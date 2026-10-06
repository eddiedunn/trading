"""Unit tests for the strategy agent: validation, feedback, the loop,
and the hand-off. The Anthropic client and the backtest API are mocked."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent import backtest_client as api
from agent.cli import main
from agent.llm import ReplyError, StrategyWriter, extract_code
from agent.loop import Config, final_test_command, run
from agent.prompts import system_prompt
from agent.runlog import RunLog
from agent.validate import strategy_name_from_code, validate_strategy

EXAMPLE = (Path(__file__).resolve().parent.parent / "strategies" / "examples" / "EmaCross.py").read_text()

GOOD = EXAMPLE.replace("EmaCross", "Good")


def _phase1(passed=True):
    return {"phase": 1, "passed": passed, "campaign": "2026-04-06", "attempts": 7, "required_sharpe": 1.32, "stats": {
        "total_return": 0.3, "sharpe": 1.0, "max_drawdown": -0.1, "win_rate": 0.5, "profit_factor": 1.5,
        "trade_count": 40, "calmar": 2.0, "cagr": 0.12, "years": 2.3,
        "benchmark": {"total_return": 0.8, "sharpe": 0.71, "max_drawdown": -0.55},
        "alpha_beta": {"beta": 0.42, "alpha_annual": 0.05},
        "floor_failures": ["sharpe below required"] if not passed else [],
        "gate": {"sharpe": {"value": 1.0, "threshold": 1.32, "passed": passed},
                 "profit_factor": {"value": 1.5, "threshold": 1.3, "passed": True}},
        "per_pair": {"BTC_USDC-USDC_4h": {"total_return": 0.3, "profit_factor": 1.5, "max_drawdown": -0.1, "trade_count": 40}}}}


def _phase2(passed=True):
    return {"phase": 2, "passed": passed, "windows": [
        {"label": "period 1", "timerange": "20240101-20240901", "passed": True, "trades": 20, "profit_total": 0.1,
         "profit_factor": 1.4, "max_drawdown": -0.1},
        {"label": "period 2", "timerange": "20240901-20250501", "passed": True, "trades": 18, "profit_total": 0.07,
         "profit_factor": 1.3, "max_drawdown": -0.12},
        {"label": "period 3", "timerange": "20250501-20260101", "passed": passed, "trades": 15, "profit_total": -0.04,
         "profit_factor": 0.9, "max_drawdown": -0.3},
    ]}


class TestValidate:
    def test_example_in_right_name_passes(self):
        assert validate_strategy(GOOD, "Good") == []
        assert strategy_name_from_code(GOOD) == "Good"

    def test_wrong_class_name(self):
        assert any("must be named 'Other'" in p for p in validate_strategy(GOOD, "Other"))

    def test_syntax_error(self):
        assert validate_strategy("def generate_signals(df:\n", "X")[0].startswith("File does not parse")

    def test_missing_generate_signals(self):
        code = GOOD.replace("def generate_signals", "def make_signals")
        assert "Missing module-level function generate_signals(df)" in validate_strategy(code, "Good")

    @pytest.mark.parametrize("line", ["import os", "import subprocess", "from socket import socket",
                                      "import requests", "from . import x"])
    def test_banned_imports(self, line):
        problems = validate_strategy(line + "\n" + GOOD, "Good")
        assert any("not allowed" in p for p in problems)

    def test_freqtrade_import_in_try_is_allowed(self):
        assert "freqtrade" in GOOD and validate_strategy(GOOD, "Good") == []

    def test_banned_calls_and_lookahead(self):
        code = GOOD.replace("return (fast > slow).astype(int)",
                            "open('x'); return (fast > slow).astype(int).shift(-1)")
        problems = validate_strategy(code, "Good")
        assert any("open()" in p for p in problems)
        assert any("looks ahead" in p for p in problems)

    def test_too_many_constants(self):
        code = "A = 1\nB = 2\nC = 3\nD = 4\nE = 5\n" + GOOD  # 2-5 plus 12, 26, 100 and -0.05 = 8 (1 is exempt)
        assert any("tunable numbers" in p for p in validate_strategy(code, "Good"))
        assert validate_strategy(code, "Good", max_params=8) == []

    def test_bad_name(self):
        assert any("identifier" in p for p in validate_strategy(GOOD, "Bad-Name"))

    def test_example_file_itself_validates_clean(self):
        assert validate_strategy(EXAMPLE, "EmaCross") == []


# EmaCross carries three tunable numbers (12, 26 and minimal_roi's 100), so with
# max_params=3 any one extra number pushes it over the cap.
_SIGNAL_LINE = "    return (fast > slow).astype(int)"


def _with_module_line(line):
    return GOOD.replace("FAST = 12\n", f"FAST = 12\n{line}\n")


def _with_signal_expr(expr):
    return GOOD.replace(_SIGNAL_LINE, f"    extra = {expr}\n{_SIGNAL_LINE}")


def _over_cap(code, max_params=4):
    return [p for p in validate_strategy(code, "Good", max_params=max_params) if "tunable numbers" in p]


class TestParamCap:
    def test_example_is_at_four(self):  # 12, 26, minimal_roi 100, stoploss -0.05
        assert _over_cap(GOOD) == []

    @pytest.mark.parametrize("line", [
        "A1 = -0.07",
        "A2: int = 7",
        "A3 = 3 * 4",
        "A4, A5 = 10, 20",
        "lookback = 50",
        "WINDOWS = (2, 3, 5, 8, 13, 21, 34, 55)",
        "LEVELS = [7.5]",
    ])
    def test_module_level_bypasses_are_counted(self, line):
        assert _over_cap(_with_module_line(line)), line

    @pytest.mark.parametrize("expr", [
        'df["close"].rolling(20).mean()',
        'df["close"] > 1.5',
        'df["close"].ewm(span=200).mean()',
        'df["close"] * -3',
    ])
    def test_inline_literals_in_functions_are_counted(self, expr):
        assert _over_cap(_with_signal_expr(expr)), expr

    def test_eight_item_tuple_is_eight_knobs(self):
        problems = _over_cap(_with_module_line("WINDOWS = (2, 3, 5, 8, 13, 21, 34, 55)"), max_params=6)
        assert problems and problems[0].startswith("12 distinct tunable numbers")
        assert "55 (line" in problems[0]

    def test_exempt_values_and_reuse_do_not_count(self):
        code = _with_signal_expr('df["close"].shift(1) * 0 + df["close"].rolling(FAST).mean() - 1 + 12 + -1')
        assert _over_cap(code) == []

    def test_negative_is_distinct_from_positive(self):
        assert _over_cap(_with_signal_expr("-12")), "-12 is a different knob from 12"

    def test_boilerplate_class_attrs_are_exempt(self):
        code = GOOD.replace("startup_candle_count = SLOW * 3",
                            "startup_candle_count = SLOW * 3\n    INTERFACE_VERSION = 3")
        assert _over_cap(code) == []

    def test_stops_are_knobs(self):
        """No config overrides the strategy's stops any more, so they count toward the cap."""
        code = GOOD.replace("startup_candle_count = SLOW * 3",
                            "startup_candle_count = SLOW * 3\n    trailing_stop_positive = 0.02\n"
                            "    trailing_stop_positive_offset = 0.03\n    stoploss = -0.07")
        assert _over_cap(code)

    def test_minimal_roi_is_not_exempt(self):
        code = GOOD.replace('minimal_roi = {"0": 100}', 'minimal_roi = {"0": 100, "60": 0.04}')
        assert _over_cap(code)

    def test_problem_message_tells_claude_what_counts(self):
        problems = _over_cap(_with_module_line("A4, A5 = 10, 20"))
        assert "Every numeric literal anywhere in the file counts except 0, 1 and -1" in problems[0]
        assert "10 (line" in problems[0] and "20 (line" in problems[0]


class TestLookAhead:
    @pytest.mark.parametrize("expr", [
        'df["close"].shift(-1)',
        'df["close"].shift(periods=-2)',
        'df["close"].shift(BACK)',
        'df["close"].shift(-FAST)',
        'df["close"].shift(len(df) - 2000)',
        'df["close"].shift(FAST - 30)',
        'df["close"].diff(-1)',
        'df["close"].diff(periods=-3)',
        'df["close"].pct_change(-1)',
        'df["close"].pct_change(periods=-4)',
        'df["close"].rolling(FAST, center=True).mean()',
        'df["close"].bfill()',
        'df["close"].backfill()',
        'df["close"].fillna(method="bfill")',
        'df["close"].fillna(method="backfill")',
    ])
    def test_rejected(self, expr):
        code = _with_module_line("BACK = -1").replace(
            _SIGNAL_LINE, f"    extra = {expr}\n{_SIGNAL_LINE}")
        problems = validate_strategy(code, "Good", max_params=20)
        assert any("look" in p or "future" in p for p in problems), (expr, problems)

    @pytest.mark.parametrize("expr", [
        'df["close"].shift()',
        'df["close"].shift(1)',
        'df["close"].shift(FAST)',
        'df["close"].shift(periods=SLOW)',
        'df["close"].diff()',
        'df["close"].pct_change(FAST)',
        'df["close"].rolling(FAST, center=False).mean()',
        'df["close"].ffill()',
        'df["close"].fillna(0)',
    ])
    def test_allowed(self, expr):
        assert validate_strategy(_with_signal_expr(expr), "Good", max_params=20) == []

    def test_constant_reassigned_elsewhere_is_not_trusted(self):
        code = GOOD.replace(_SIGNAL_LINE, f"    FAST = -1\n    extra = df.shift(FAST)\n{_SIGNAL_LINE}")
        assert any("shift() period" in p for p in validate_strategy(code, "Good", max_params=20))


class TestFeedback:
    def test_phase2_feedback_shows_every_period_in_full(self):
        """All three periods are development data now, so nothing is hidden."""
        text = api.phase2_feedback(_phase2(passed=False))
        assert "period 1 (20240101-20240901): passed (trades=20, profit_total=0.1000, pf=1.4000, dd=-0.1000)" in text
        assert "period 3 (20250501-20260101): failed (trades=15, profit_total=-0.0400, pf=0.9000, dd=-0.3000)" in text

    def test_phase2_feedback_shows_window_error(self):
        result = {"passed": False, "windows": [{"label": "period 1", "passed": False, "error": "boom"}]}
        assert "period 1: failed — error: boom" in api.phase2_feedback(result)

    def test_phase1_feedback_has_gate_and_pairs(self):
        text = api.phase1_feedback(_phase1(False))
        assert "Phase 1 FAILED" in text and "profit_factor=1.5000" in text and "BTC_USDC-USDC_4h" in text

    def test_phase1_feedback_has_gate_breakdown_benchmark_and_attempts(self):
        text = api.phase1_feedback(_phase1(False))
        assert "sharpe: 1.0000 vs threshold 1.3200 — FAILED" in text
        assert "profit_factor: 1.5000 vs threshold 1.3000 — passed" in text
        assert "Campaign attempts so far: 7 effective" in text and "Phase 1 Sharpe bar is 1.3200" in text
        assert "Strategy sharpe=1.0000 vs buy-and-hold 0.7100" in text and "beta=0.4200" in text
        assert "Floor failures: sharpe below required" in text

    def test_phase1_feedback_copes_with_old_shape(self):
        text = api.phase1_feedback({"passed": True, "stats": {"sharpe": 1.0}})
        assert "Phase 1 PASSED" in text and "Gate checks" not in text and "Strategy sharpe=" not in text


class TestBacktestClient:
    @patch("agent.backtest_client.httpx.post")
    def test_posts_schema_and_returns_json(self, post):
        post.return_value = httpx.Response(200, json=_phase1(), request=httpx.Request("POST", "http://x"))
        assert api.run_phase("S", "code", 1, "http://api/")["passed"] is True
        kwargs = post.call_args.kwargs
        assert post.call_args.args[0] == "http://api/backtest"
        assert kwargs["json"] == {"strategy_name": "S", "strategy_code": "code", "phase": 1}
        assert kwargs["timeout"] == api.PHASE1_TIMEOUT

    @patch("agent.backtest_client.httpx.post")
    def test_error_detail_becomes_feedback(self, post):
        post.return_value = httpx.Response(500, json={"detail": "KeyError: 'close'"}, request=httpx.Request("POST", "http://x"))
        with pytest.raises(api.BacktestError, match="HTTP 500.*KeyError"):
            api.run_phase("S", "code", 2)


class TestLLM:
    def test_extract_code_takes_last_fenced_block(self):
        assert extract_code("x\n```python\na = 1\n```\n```\nb = 2\n```") == "b = 2\n"
        with pytest.raises(ReplyError):
            extract_code("no code here")

    def test_system_prompt_carries_example_and_rules(self):
        text = system_prompt()
        assert "class EmaCross(IStrategy)" in text
        assert "Do not tune to the reported numbers" in text
        assert "timestamp, open, high, low, close, volume" in text
        assert "date, open" not in text
        assert "At most 6 distinct tunable numbers" in text
        for banned in ("rolling(..., center=True)", "bfill()", "pct_change"):
            assert banned in text

    def test_system_prompt_describes_the_extra_data(self):
        text = system_prompt()
        for col in ("funding_rate", "close_BTC", "close_ETH", "close_SOL",
                    "funding_BTC", "funding_ETH", "funding_SOL"):
            assert col in text
        assert "stamped T+1h .. T+4h" in text  # the alignment
        assert "NaN where there is no funding data" in text
        assert 'get_pair_dataframe(metadata["pair"], "1h", candle_type="funding_rate")' in text
        assert "merge_informative_pair" in text and "def informative_pairs" in text
        assert "can_short = True" in text
        assert "-35%" in text and "-20%" in text
        assert "startup_candle_count` at 120" in text

    def test_default_seeds_are_varied(self):
        from agent.prompts import DEFAULT_SEEDS
        assert 8 <= len(DEFAULT_SEEDS) <= 10
        joined = " ".join(DEFAULT_SEEDS).lower()
        for idea in ("funding", "carry", "relative-strength", "lead-lag", "volatility regime",
                     "short", "market-neutral", "liquidation"):
            assert idea in joined

    @staticmethod
    def _cli_reply(text="```python\nx = 1\n```", **extra):
        body = {"is_error": False, "stop_reason": "end_turn", "session_id": "sess-1", "result": text,
                "usage": {"input_tokens": 10, "output_tokens": 5}, "total_cost_usd": 0.01, **extra}
        return SimpleNamespace(returncode=0, stdout=json.dumps(body), stderr="")

    def test_writer_resumes_session_and_passes_model(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-reach-claude")
        runner = MagicMock(return_value=self._cli_reply())
        w = StrategyWriter(model="claude-test", runner=runner)
        assert w.write("idea", "Name") == "x = 1\n"
        assert w.revise("feedback") == "x = 1\n"
        first, second = runner.call_args_list
        assert first.args[0][first.args[0].index("--model") + 1] == "claude-test"
        assert "--resume" not in first.args[0]
        assert second.args[0][-2:] == ["--resume", "sess-1"]
        assert "feedback" in second.kwargs["input"]
        assert first.args[0][first.args[0].index("--tools") + 1] == ""
        assert "ANTHROPIC_API_KEY" not in first.kwargs["env"]
        assert w.usage == {"input_tokens": 20, "output_tokens": 10, "cost_usd": 0.02}

    def test_writer_refusal(self):
        runner = MagicMock(return_value=self._cli_reply(stop_reason="refusal"))
        with pytest.raises(ReplyError, match="declined"):
            StrategyWriter(runner=runner).write("idea", "Name")

    def test_writer_cli_failure(self):
        runner = MagicMock(return_value=SimpleNamespace(returncode=1, stdout="", stderr="not logged in"))
        with pytest.raises(ReplyError, match="not logged in"):
            StrategyWriter(runner=runner).write("idea", "Name")


class FakeWriter:
    """Returns scripted replies; records the feedback it was given."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.feedback = []
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def write(self, seed, name):
        self.name = name
        return self.replies.pop(0)(name)

    def revise(self, feedback):
        self.feedback.append(feedback)
        return self.replies.pop(0)(self.name)


def good(name):
    return EXAMPLE.replace("EmaCross", name)


def with_os(name):
    return "import os\n" + good(name)


def _config(tmp_path, **kw):
    return Config(seeds=["idea"], max_strategies=1, max_iterations=3, max_phase2=1, api_url="http://api", **kw)


def _log(tmp_path):
    return RunLog(runs_dir=tmp_path, run_id="run1")


def _entries(runlog):
    return [json.loads(l) for l in runlog.path.read_text().splitlines()]


class TestLoop:
    @patch("agent.loop.api.run_phase")
    def test_validation_failure_is_fed_back_then_passes(self, run_phase, tmp_path, capsys):
        run_phase.side_effect = [_phase1(), _phase2()]
        writer = FakeWriter([with_os, good])
        runlog = _log(tmp_path)

        outcomes = run(_config(tmp_path), runlog, writer_factory=lambda: writer)

        assert outcomes[0].passed and outcomes[0].attempts == 2
        assert "Import of 'os' is not allowed" in writer.feedback[0]
        name = outcomes[0].name
        assert (tmp_path / "run1" / f"{name}.py").exists()
        assert json.loads((tmp_path / "run1" / f"{name}.phase2.json").read_text())["passed"] is True
        stages = [(e["stage"], e["passed"]) for e in _entries(runlog)]
        assert stages == [("llm", True), ("validate", False), ("llm", True), ("validate", True),
                          ("phase1", True), ("phase2", True), ("result", True)]
        out = capsys.readouterr().out
        assert f"scripts/submit_strategy.sh {tmp_path / 'run1' / name}.py" in out
        assert "paper-add" not in out and "--force" not in out
        assert all(c.args[2] in (1, 2) for c in run_phase.call_args_list)

    @patch("agent.loop.api.run_phase")
    def test_phase1_failure_feedback_and_iteration_cap(self, run_phase, tmp_path):
        run_phase.return_value = _phase1(passed=False)
        writer = FakeWriter([good, good, good])

        outcomes = run(_config(tmp_path), _log(tmp_path), writer_factory=lambda: writer)

        assert not outcomes[0].passed and outcomes[0].attempts == 3
        assert outcomes[0].reason == "iteration budget used up"
        assert all("Phase 1 FAILED" in f for f in writer.feedback)
        assert all(c.kwargs == {} and c.args[2] == 1 for c in run_phase.call_args_list)  # never Phase 2

    @patch("agent.loop.api.run_phase")
    def test_phase2_budget_stops_the_strategy(self, run_phase, tmp_path):
        run_phase.side_effect = [_phase1(), _phase2(passed=False), _phase1()]
        writer = FakeWriter([good, good])

        outcomes = run(_config(tmp_path), _log(tmp_path), writer_factory=lambda: writer)

        assert outcomes[0].reason == "Phase 2 budget used up"
        assert outcomes[0].phase2_runs == 1
        assert "period 3 (20250501-20260101): failed" in writer.feedback[0] and "pf=0.9000" in writer.feedback[0]
        assert "Phase 1 PASSED" in writer.feedback[0]

    @patch("agent.loop.api.run_phase")
    def test_api_error_is_fed_back(self, run_phase, tmp_path):
        run_phase.side_effect = [api.BacktestError("Phase 1 could not run (HTTP 500): KeyError"), _phase1(), _phase2()]
        writer = FakeWriter([good, good])

        outcomes = run(_config(tmp_path), _log(tmp_path), writer_factory=lambda: writer)

        assert outcomes[0].passed and "KeyError" in writer.feedback[0]

    @patch("agent.loop.api.run_phase")
    def test_unusable_reply_ends_strategy(self, run_phase, tmp_path):
        writer = MagicMock()
        writer.write.side_effect = ReplyError("no code")

        outcomes = run(_config(tmp_path), _log(tmp_path), writer_factory=lambda: writer)

        assert not outcomes[0].passed and "unusable" in outcomes[0].reason
        run_phase.assert_not_called()

    def test_strategy_names_are_unique_identifiers(self, tmp_path):
        with patch("agent.loop.api.run_phase", return_value=_phase1(False)):
            cfg = Config(max_strategies=2, max_iterations=1, api_url="http://api")
            outcomes = run(cfg, _log(tmp_path), writer_factory=lambda: FakeWriter([good]))
        names = [o.name for o in outcomes]
        assert len(set(names)) == 2 and all(api_name_ok(n) for n in names)


def api_name_ok(name):
    from backtest_api.main import BacktestRequest
    return BacktestRequest(strategy_name=name, strategy_code="", phase=1).strategy_name == name


class TestHandOff:
    def test_command_is_submit_script_never_force(self, tmp_path):
        paths = {"code": tmp_path / "S.py", "phase1": tmp_path / "S.phase1.json", "phase2": tmp_path / "S.phase2.json"}
        assert final_test_command(paths) == f"scripts/submit_strategy.sh {tmp_path / 'S.py'}"


class TestNoFinalTest:
    """The agent must never see the held-back data: no path to phase 3."""

    @patch("agent.backtest_client.httpx.post")
    def test_run_phase_refuses_phase_3(self, post):
        with pytest.raises(ValueError, match="final test"):
            api.run_phase("S", "code", 3)
        post.assert_not_called()

    def test_no_phase_3_in_agent_source(self):
        import agent
        for path in Path(agent.__file__).parent.glob("*.py"):
            src = path.read_text()
            assert "run_phase(name, code, 3" not in src and '"phase": 3' not in src, path
        assert api.AGENT_PHASES == (1, 2)

    def test_no_paper_add_code_path(self):
        """Paper now needs the final test first, so the agent no longer queues anything."""
        import agent.cli, agent.loop
        for mod in (agent.cli, agent.loop):
            src = Path(mod.__file__).read_text()
            assert "trading_client" not in src and "subprocess" not in src and "queue_paper" not in src


class TestCli:
    @patch("agent.cli.run")
    @patch("agent.cli.RunLog")
    def test_run_passes_options(self, runlog, run_mock, monkeypatch):
        run_mock.return_value = [SimpleNamespace(passed=False)]
        rc = main(["run", "--seed", "a", "--seed", "b", "--max-strategies", "3", "--max-phase2", "1",
                   "--model", "m", "--api-url", "http://a"])
        assert rc == 1
        cfg = run_mock.call_args.args[0]
        assert cfg.seeds == ["a", "b"] and cfg.max_strategies == 3 and cfg.max_phase2 == 1
        assert cfg.model == "m" and cfg.api_url == "http://a"

    def test_queue_paper_flag_is_gone(self):
        with pytest.raises(SystemExit):
            main(["run", "--queue-paper"])

    def test_no_promote_code_path(self):
        import agent.cli, agent.loop
        for mod in (agent.cli, agent.loop):
            src = Path(mod.__file__).read_text()
            assert '"promote"' not in src and "trading_client import" not in src
