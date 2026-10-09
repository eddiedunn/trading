"""Tests for the paper-vs-backtest comparison (paper/compare.py). No network or podman."""

import json
import math
import zipfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from paper.compare import (
    PAIRS,
    TOLERANCES,
    candle_floor,
    compare,
    evaluate_paper_run,
    prepare_data,
    profit_factor,
    run_comparison_backtest,
    save_result,
    summarize_backtest,
    summarize_paper,
)

START = datetime(2026, 8, 1, 5, 30, tzinfo=timezone.utc).timestamp()
END = datetime(2026, 9, 15, 10, 0, tzinfo=timezone.utc).timestamp()


@pytest.fixture(autouse=True)
def paper_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_PAPER_DIR", str(tmp_path))
    monkeypatch.delenv("TRADING_PAPER_FT_DATA_DIR", raising=False)
    monkeypatch.delenv("TRADING_PAPER_DATA_DIR", raising=False)
    return tmp_path


def _paper(n=20, ret=5.0, pf=1.5, worst=-4.0):
    return {"trade_count": n, "return_pct": ret, "profit_factor": pf, "worst_trade_pct": worst}


def _bt(n=20, ret=6.0, pf=1.6, stoploss=-0.05):
    return {"trade_count": n, "return_pct": ret, "profit_factor": pf, "stoploss": stoploss}


class TestCandleFloor:
    def test_floors_to_4h_boundary(self):
        assert candle_floor(START) == datetime(2026, 8, 1, 4, tzinfo=timezone.utc).timestamp()
        assert candle_floor(END) == datetime(2026, 9, 15, 8, tzinfo=timezone.utc).timestamp()


class TestProfitFactor:
    def test_ratio(self):
        assert profit_factor([3.0, -1.0, -1.0]) == 1.5

    def test_no_losses_is_infinite(self):
        assert profit_factor([1.0, 2.0]) == math.inf

    def test_no_trades_is_zero(self):
        assert profit_factor([]) == 0.0


class TestCompare:
    def test_matching_run_passes_and_reports_every_check(self):
        out = compare(_paper(), _bt())

        assert out["passed"] is True
        assert set(out["checks"]) == {"trade_count", "return_pct", "worst_trade_pct", "profit_factor"}
        for c in out["checks"].values():
            assert {"value", "threshold", "passed"} <= set(c)

    def test_trade_count_within_40_percent(self):
        assert compare(_paper(n=28), _bt(n=20))["checks"]["trade_count"]["passed"] is True
        assert compare(_paper(n=29), _bt(n=20))["checks"]["trade_count"]["passed"] is False
        assert compare(_paper(n=12), _bt(n=20))["checks"]["trade_count"]["passed"] is True
        assert compare(_paper(n=11), _bt(n=20))["checks"]["trade_count"]["passed"] is False

    def test_trade_count_within_3_when_small(self):
        assert compare(_paper(n=5), _bt(n=2))["checks"]["trade_count"]["passed"] is True
        assert compare(_paper(n=6), _bt(n=2))["checks"]["trade_count"]["passed"] is False
        assert compare(_paper(n=0), _bt(n=2))["checks"]["trade_count"]["threshold"] == [0.0, 5]

    def test_return_floor_uses_2pp_for_small_backtest_returns(self):
        check = compare(_paper(ret=-1.0), _bt(ret=1.0))["checks"]["return_pct"]
        assert check["threshold"] == -1.0 and check["passed"] is True
        assert compare(_paper(ret=-1.01), _bt(ret=1.0))["checks"]["return_pct"]["passed"] is False

    def test_return_floor_uses_half_of_large_backtest_returns(self):
        assert compare(_paper(ret=10.0), _bt(ret=20.0))["checks"]["return_pct"]["threshold"] == 10.0
        # a losing backtest: paper may lose up to 50% more
        assert compare(_paper(ret=-15.0), _bt(ret=-10.0))["checks"]["return_pct"]["threshold"] == -15.0
        assert compare(_paper(ret=30.0), _bt(ret=5.0))["checks"]["return_pct"]["passed"] is True

    def test_worst_trade_beyond_stop_plus_1pp_fails(self):
        assert compare(_paper(worst=-6.0), _bt(stoploss=-0.05))["checks"]["worst_trade_pct"]["passed"] is True
        check = compare(_paper(worst=-6.5), _bt(stoploss=-0.05))["checks"]["worst_trade_pct"]
        assert check["threshold"] == -6.0 and check["passed"] is False

    def test_no_paper_trades_passes_worst_trade(self):
        assert compare(_paper(n=0, worst=None), _bt(n=0))["checks"]["worst_trade_pct"]["passed"] is True

    def test_missing_stoploss_fails(self):
        assert compare(_paper(), _bt(stoploss=None))["passed"] is False

    def test_profit_factor_checked_only_with_10_backtest_trades(self):
        assert compare(_paper(n=10, pf=1.0), _bt(n=10, pf=1.5))["checks"]["profit_factor"]["passed"] is False
        assert compare(_paper(n=10, pf=1.2), _bt(n=10, pf=1.5))["checks"]["profit_factor"]["passed"] is True
        skipped = compare(_paper(n=9, pf=0.1), _bt(n=9, pf=1.5))["checks"]["profit_factor"]
        assert skipped["passed"] is True and skipped["threshold"] is None and "note" in skipped

    def test_profit_factor_with_no_losses(self):
        assert compare(_paper(pf=math.inf), _bt(pf=math.inf))["checks"]["profit_factor"]["passed"] is True
        assert compare(_paper(pf=3.0), _bt(pf=math.inf))["checks"]["profit_factor"]["passed"] is False


class TestSummaries:
    def test_backtest(self):
        bt = {"total_trades": 3, "profit_total": 0.042, "stoploss": -0.08,
              "trades": [{"profit_abs": 3.0}, {"profit_abs": -1.0}, {"profit_abs": 1.0}]}
        assert summarize_backtest(bt) == {"trade_count": 3, "return_pct": pytest.approx(4.2),
                                          "profit_factor": 4.0, "stoploss": -0.08}

    def test_paper_includes_open_trades(self):
        metrics = {"trade_count": 3, "closed_trade_count": 2, "profit_pct": 1.5}
        trades = [{"profit_ratio": 0.04, "profit_abs": 2.0}, {"profit_ratio": -0.03, "profit_abs": -1.0},
                  {"profit_ratio": -0.07, "profit_abs": -1.0}]  # last one still open
        s = summarize_paper(metrics, trades)
        assert s["worst_trade_pct"] == pytest.approx(-7.0)
        assert s["profit_factor"] == 1.0
        assert s["return_pct"] == 1.5 and s["trade_count"] == 3


class TestPrepareData:
    @patch("paper.compare.export_freqtrade")
    @patch("paper.compare.download_funding")
    @patch("paper.compare.download_pair")
    @patch("paper.compare.ccxt.hyperliquid")
    def test_downloads_from_warmup_and_checks_coverage(
        self, mock_ex, mock_pair, mock_funding, mock_export, paper_dir
    ):
        ft = paper_dir / "ftdata" / "hyperliquid" / "futures"

        def export(pair, raw, ftdir):
            ft.mkdir(parents=True, exist_ok=True)
            dates = pd.date_range("2026-05-01", "2026-09-15 04:00", freq="4h", tz="UTC")
            pd.DataFrame({"date": dates}).to_feather(ft / f"{pair.replace('/', '_').replace(':', '_')}-4h-futures.feather")

        mock_export.side_effect = export
        stale = paper_dir / "data" / "BTC_USDC-USDC_4h.feather"
        stale.parent.mkdir(parents=True)
        stale.write_text("old")

        prepare_data(START, END, pairs=["BTC/USDC:USDC"])

        assert not stale.exists()  # re-downloaded so no half-formed candle lingers
        args = mock_pair.call_args.args
        assert args[1:3] == ("BTC/USDC:USDC", "4h") and args[4] == "2026-05-03"
        assert mock_funding.call_args.args[3] == "2026-05-03"
        assert mock_export.call_args.args[2] == paper_dir / "ftdata"

    @patch("paper.compare.export_freqtrade")
    @patch("paper.compare.download_funding")
    @patch("paper.compare.download_pair")
    @patch("paper.compare.ccxt.hyperliquid")
    def test_raises_when_candles_stop_short(self, mock_ex, mock_pair, mock_funding, mock_export, paper_dir):
        ft = paper_dir / "ftdata" / "hyperliquid" / "futures"

        def export(pair, raw, ftdir):
            ft.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"date": pd.date_range("2026-05-01", "2026-09-01", freq="4h", tz="UTC")}).to_feather(
                ft / "BTC_USDC_USDC-4h-futures.feather")

        mock_export.side_effect = export
        with pytest.raises(RuntimeError, match="candles end at"):
            prepare_data(START, END, pairs=["BTC/USDC:USDC"])


def _write_result(out_dir, strategy, stats):
    name = "backtest-result-2026.zip"
    with zipfile.ZipFile(out_dir / name, "w") as zf:
        zf.writestr("backtest-result-2026.json", json.dumps({"strategy": {strategy: stats}}))
    (out_dir / ".last_result.json").write_text(json.dumps({"latest_backtest": name}))


class TestRunComparisonBacktest:
    @patch("paper.compare.subprocess.run")
    def test_runs_freqtrade_over_the_paper_period(self, mock_run, paper_dir):
        timerange = f"{candle_floor(START)}-{candle_floor(END)}"
        out_dir = paper_dir / "compare" / "StratA" / timerange

        def fake_run(cmd, **kw):
            _write_result(out_dir, "StratA", {"total_trades": 4})
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = fake_run

        stats = run_comparison_backtest("StratA", START, END)

        assert stats == {"total_trades": 4}
        cmd = mock_run.call_args.args[0]
        assert cmd[:3] == ["podman", "run", "--rm"]
        assert "--userns=keep-id:uid=1000,gid=1000" in cmd
        assert cmd[cmd.index("--timerange") + 1] == timerange
        assert cmd[cmd.index("--strategy") + 1] == "StratA"
        assert f"{paper_dir / 'strategies'}:/freqtrade/strategies:ro,Z" in cmd
        assert f"{paper_dir / 'ftdata'}:/freqtrade/user_data/data:ro,Z" in cmd
        from paper.orchestrator import FREQTRADE_IMAGE
        assert FREQTRADE_IMAGE in cmd
        config = json.loads((out_dir / "config.json").read_text())
        assert config["stake_amount"] == "unlimited" and config["max_open_trades"] == len(PAIRS)

    @patch("paper.compare.subprocess.run", return_value=MagicMock(returncode=2, stderr="no data for pair"))
    def test_freqtrade_error_raises_with_stderr(self, mock_run):
        with pytest.raises(RuntimeError, match="no data for pair"):
            run_comparison_backtest("StratA", START, END)

    @patch("paper.compare.subprocess.run", return_value=MagicMock(returncode=0, stderr=""))
    def test_missing_result_raises(self, mock_run):
        with pytest.raises(RuntimeError, match="no result"):
            run_comparison_backtest("StratA", START, END)


class TestEvaluatePaperRun:
    METRICS = {"trade_count": 12, "closed_trade_count": 12, "profit_pct": 4.0}
    TRADES = [{"profit_ratio": 0.03, "profit_abs": 3.0}] * 9 + [{"profit_ratio": -0.02, "profit_abs": -2.0}] * 3

    @patch("paper.compare.run_comparison_backtest")
    @patch("paper.compare.prepare_data")
    def test_pass(self, mock_prep, mock_bt):
        mock_bt.return_value = {"total_trades": 11, "profit_total": 0.05, "stoploss": -0.05,
                                "trades": [{"profit_abs": 3.0}] * 8 + [{"profit_abs": -2.0}] * 3}

        r = evaluate_paper_run("StratA", START, END, self.METRICS, self.TRADES)

        assert r["passed"] is True and r["reason"] == "matches backtest"
        mock_prep.assert_called_once_with(START, END)
        mock_bt.assert_called_once_with("StratA", START, END)
        assert r["backtest"]["trade_count"] == 11 and r["paper"]["trade_count"] == 12

    @patch("paper.compare.run_comparison_backtest")
    @patch("paper.compare.prepare_data")
    def test_fail_names_failed_checks(self, mock_prep, mock_bt):
        mock_bt.return_value = {"total_trades": 30, "profit_total": 0.05, "stoploss": -0.05,
                                "trades": [{"profit_abs": 1.0}] * 30}

        r = evaluate_paper_run("StratA", START, END, self.METRICS, self.TRADES)

        assert r["passed"] is False and "trade_count" in r["reason"]

    @patch("paper.compare.run_comparison_backtest")
    @patch("paper.compare.prepare_data", side_effect=RuntimeError("Hyperliquid 503"))
    def test_backtest_that_cannot_run_is_a_fail_with_reason(self, mock_prep, mock_bt):
        r = evaluate_paper_run("StratA", START, END, self.METRICS, self.TRADES)

        assert r["passed"] is False
        assert "comparison backtest could not run" in r["reason"] and "503" in r["reason"]
        mock_bt.assert_not_called()

    @patch("paper.compare.prepare_data")
    def test_no_metrics_is_a_fail(self, mock_prep):
        r = evaluate_paper_run("StratA", START, END, None, None)

        assert r["passed"] is False and "no final paper metrics" in r["reason"]
        mock_prep.assert_not_called()

    def test_save_result(self, paper_dir):
        r = {"strategy": "StratA", "timerange": "1-2", "passed": True,
             "checks": {"profit_factor": {"value": math.inf}}}
        path = save_result(r)
        assert path == paper_dir / "compare" / "StratA" / "comparison-1-2.json"
        assert json.loads(path.read_text())["passed"] is True


def test_tolerances_documented():
    assert TOLERANCES == {
        "trade_count_rel": 0.40, "trade_count_abs": 3,
        "return_abs_pp": 2.0, "return_rel": 0.50,
        "stop_slippage_pp": 1.0,
        "profit_factor_rel": 0.80, "profit_factor_min_trades": 10,
    }
