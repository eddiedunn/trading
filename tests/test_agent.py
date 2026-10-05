"""Unit tests for the strategy agent: validation, feedback redaction, the loop,
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
from agent.loop import Config, paper_add_commands, queue_paper, run
from agent.prompts import system_prompt
from agent.runlog import RunLog
from agent.validate import strategy_name_from_code, validate_strategy

EXAMPLE = (Path(__file__).resolve().parent.parent / "strategies" / "examples" / "EmaCross.py").read_text()

GOOD = EXAMPLE.replace("EmaCross", "Good")


def _phase1(passed=True):
    return {"phase": 1, "passed": passed, "stats": {
        "total_return": 0.3, "sharpe": 1.0, "max_drawdown": -0.1, "win_rate": 0.5, "profit_factor": 1.5,
        "trade_count": 40, "calmar": 2.0,
        "per_pair": {"BTC_USDC-USDC_4h": {"total_return": 0.3, "profit_factor": 1.5, "max_drawdown": -0.1, "trade_count": 40}}}}


def _phase2(passed=True):
    return {"phase": 2, "passed": passed, "windows": [
        {"label": "in-sample", "timerange": "20240101-20250101", "passed": True, "profit_factor": 1.4, "max_drawdown": -0.1},
        {"label": "validation", "timerange": "20250101-20250701", "passed": True, "profit_factor": 1.3, "max_drawdown": -0.12},
        {"label": "out-of-sample", "timerange": "20250701-20260101", "passed": passed, "profit_factor": 0.9, "max_drawdown": -0.3},
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
        code = "A = 1\nB = 2\nC = 3\nD = 4\nE = 5\n" + GOOD  # plus FAST and SLOW = 7
        assert any("tunable constants" in p for p in validate_strategy(code, "Good"))
        assert validate_strategy(code, "Good", max_params=7) == []

    def test_bad_name(self):
        assert any("identifier" in p for p in validate_strategy(GOOD, "Bad-Name"))


class TestFeedback:
    def test_phase2_feedback_hides_out_of_sample_numbers_and_dates(self):
        text = api.phase2_feedback(_phase2(passed=False))
        assert "in-sample: passed (pf=1.4000, dd=-0.1000)" in text
        assert "out-of-sample: failed" in text
        assert "0.9" not in text and "-0.3" not in text
        assert "2025" not in text

    def test_phase2_feedback_shows_window_error(self):
        result = {"passed": False, "windows": [{"label": "in-sample", "passed": False, "error": "boom"}]}
        assert "in-sample: failed — error: boom" in api.phase2_feedback(result)

    def test_phase1_feedback_has_gate_and_pairs(self):
        text = api.phase1_feedback(_phase1(False))
        assert "Phase 1 FAILED" in text and "profit_factor=1.5000" in text and "BTC_USDC-USDC_4h" in text


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

    def test_writer_keeps_history_and_passes_model(self):
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            stop_reason="end_turn", usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            content=[SimpleNamespace(type="text", text="```python\nx = 1\n```")])
        w = StrategyWriter(client=client, model="claude-test")
        assert w.write("idea", "Name") == "x = 1\n"
        assert w.revise("feedback") == "x = 1\n"
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-test"
        # The same list object is passed each time (append-only history), so it now holds both turns.
        assert [m["role"] for m in w.messages] == ["user", "assistant", "user", "assistant"]
        assert kwargs["messages"] is w.messages
        assert "feedback" in w.messages[2]["content"]
        assert w.usage == {"input_tokens": 20, "output_tokens": 10}

    def test_writer_refusal(self):
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(stop_reason="refusal", usage=None, content=[])
        with pytest.raises(ReplyError, match="declined"):
            StrategyWriter(client=client).write("idea", "Name")


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
        assert "scripts/submit_strategy.sh" in out and "paper-add" in out and "--force" not in out

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
        assert "out-of-sample: failed" in writer.feedback[0] and "0.9" not in writer.feedback[0]

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

    @patch("agent.loop.queue_paper")
    @patch("agent.loop.api.run_phase")
    def test_queue_paper_flag_calls_paper_add(self, run_phase, queue, tmp_path, capsys):
        run_phase.side_effect = [_phase1(), _phase2()]
        cfg = _config(tmp_path, queue_paper=True, trinity="trin")

        outcomes = run(cfg, _log(tmp_path), writer_factory=lambda: FakeWriter([good]))

        queue.assert_called_once_with(outcomes[0].name, outcomes[0].paths, "trin")
        assert "submit_strategy.sh" not in capsys.readouterr().out

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
    def test_commands_never_force(self, tmp_path):
        paths = {"code": tmp_path / "S.py", "phase1": tmp_path / "S.phase1.json", "phase2": tmp_path / "S.phase2.json"}
        cmds = paper_add_commands("S", paths, "trinity")
        assert cmds[0] == f"scripts/submit_strategy.sh {tmp_path / 'S.py'}"
        assert "paper-add --strategy S" in cmds[1] and "--force" not in " ".join(cmds)

    @patch("agent.loop.subprocess.run")
    def test_queue_paper_runs_scp_then_paper_add_then_cleans_up(self, sp_run, tmp_path):
        paths = {"code": tmp_path / "S.py", "phase1": tmp_path / "S.phase1.json", "phase2": tmp_path / "S.phase2.json"}
        queue_paper("S", paths, "trinity")
        cmds = [c.args[0] for c in sp_run.call_args_list]
        assert cmds[0][0] == "scp" and cmds[0][-1] == "trinity:/data/services/trading/paper/"
        assert cmds[1][:2] == ["ssh", "trinity"] and "paper-add" in cmds[1] and "--force" not in cmds[1]
        assert "promote" not in " ".join(cmds[1])
        assert cmds[2][:3] == ["ssh", "trinity", "rm"]


class TestCli:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert main(["run"]) == 2

    @patch("agent.cli.run")
    @patch("agent.cli.RunLog")
    def test_run_passes_options(self, runlog, run_mock, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        run_mock.return_value = [SimpleNamespace(passed=False)]
        rc = main(["run", "--seed", "a", "--seed", "b", "--max-strategies", "3", "--max-phase2", "1",
                   "--model", "m", "--api-url", "http://a", "--queue-paper"])
        assert rc == 1
        cfg = run_mock.call_args.args[0]
        assert cfg.seeds == ["a", "b"] and cfg.max_strategies == 3 and cfg.max_phase2 == 1
        assert cfg.model == "m" and cfg.api_url == "http://a" and cfg.queue_paper is True

    def test_no_promote_code_path(self):
        import agent.cli, agent.loop
        for mod in (agent.cli, agent.loop):
            src = Path(mod.__file__).read_text()
            assert '"promote"' not in src and "trading_client import" not in src
