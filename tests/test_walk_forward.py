"""Unit tests for Phase 2 walk-forward validation.

Tests the development periods, the per-period gates, the range check and
Freqtrade result parsing.
"""

import json
import zipfile
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest_api.walk_forward import (
    run_freqtrade_backtest,
    walk_forward_test,
    _extract_profit_factor,
    _extract_max_drawdown,
    MIN_PROFIT_FACTOR,
    MAX_DRAWDOWN,
    MIN_TRADES,
    CANDLE,
    PAIRS,
    WARMUP,
    common_candle_range,
    default_windows,
    ft_timerange,
)


def _write_freqtrade_result(out_dir: Path, result_json: dict, stamp: str = "2026-10-02_21-17-10"):
    """Lay out a result the way current Freqtrade does: a zip named in .last_result.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"backtest-result-{stamp}"
    with zipfile.ZipFile(out_dir / f"{name}.zip", "w") as zf:
        zf.writestr(f"{name}.json", json.dumps(result_json))
        zf.writestr(f"{name}_config.json", "{}")
    (out_dir / ".last_result.json").write_text(json.dumps({"latest_backtest": f"{name}.zip"}))


class TestExtractMetrics:
    """Test extraction of metrics from Freqtrade JSON results."""

    def test_extract_profit_factor_nested(self):
        """Extract profit factor from nested strategy key."""
        stats = {
            "strategy": {
                "MyStrat": {
                    "profit_factor": 1.5,
                    "max_drawdown": -0.10,
                }
            }
        }
        assert _extract_profit_factor(stats) == 1.5

    def test_extract_profit_factor_direct(self):
        """Extract profit factor from direct key (fallback)."""
        stats = {"profit_factor": 1.8}
        assert _extract_profit_factor(stats) == 1.8

    def test_extract_profit_factor_missing(self):
        """Return 0 if profit factor not found."""
        stats = {}
        assert _extract_profit_factor(stats) == 0

    def test_extract_profit_factor_malformed(self):
        """Return 0 if stats is malformed."""
        assert _extract_profit_factor(None) == 0
        assert _extract_profit_factor("invalid") == 0

    def test_extract_max_drawdown_nested(self):
        """Extract max drawdown from nested strategy key."""
        stats = {
            "strategy": {
                "MyStrat": {
                    "profit_factor": 1.5,
                    "max_drawdown": -0.25,
                }
            }
        }
        assert _extract_max_drawdown(stats) == -0.25

    def test_extract_max_drawdown_account_is_negated(self):
        """Current Freqtrade reports max_drawdown_account as a positive fraction."""
        stats = {"strategy": {"MyStrat": {"max_drawdown": None, "max_drawdown_account": 0.67}}}
        assert _extract_max_drawdown(stats) == -0.67

    def test_extract_max_drawdown_direct(self):
        """Extract max drawdown from direct key (fallback)."""
        stats = {"max_drawdown": -0.15}
        assert _extract_max_drawdown(stats) == -0.15

    def test_extract_max_drawdown_missing(self):
        """Return -1 if max drawdown not found."""
        stats = {}
        assert _extract_max_drawdown(stats) == -1

    def test_extract_max_drawdown_malformed(self):
        """Return -1 if stats is malformed."""
        assert _extract_max_drawdown(None) == -1
        assert _extract_max_drawdown([]) == -1


class TestRunFreqtradeBacktest:
    """Test single backtest execution and result parsing."""

    @pytest.fixture(autouse=True)
    def _results_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADING_RESULTS_DIR", str(tmp_path / "results"))
        self.results = tmp_path / "results"

    @patch("backtest_api.walk_forward.subprocess.run")
    def test_successful_backtest(self, mock_subprocess):
        """Successful backtest run returns the JSON inside the latest result zip."""
        result_json = {
            "strategy": {"TestStrat": {"profit_factor": 1.3, "max_drawdown_account": 0.18}}
        }

        def fake_run(cmd, **kwargs):
            _write_freqtrade_result(self.results / "TestStrat" / "20230101-20240601", result_json)
            return MagicMock(returncode=0, stderr="", stdout="")

        mock_subprocess.side_effect = fake_run

        result = run_freqtrade_backtest("TestStrat", "20230101-20240601")

        assert result == result_json

    @patch("backtest_api.walk_forward.subprocess.run")
    def test_failed_backtest(self, mock_subprocess):
        """Failed backtest returns error dict."""
        mock_subprocess.return_value = MagicMock(
            returncode=1, stderr="Out of memory", stdout=""
        )

        result = run_freqtrade_backtest("TestStrat", "20230101-20240601")

        assert result is not None
        assert "error" in result
        assert result["error"] == "Out of memory"
        assert result["returncode"] == 1

    @patch("backtest_api.walk_forward.subprocess.run")
    def test_backtest_no_results_file(self, mock_subprocess):
        """Backtest succeeds but no results file → return None."""
        mock_subprocess.return_value = MagicMock(
            returncode=0, stderr="", stdout=""
        )

        result = run_freqtrade_backtest("TestStrat", "20230101-20240601")

        assert result is None

    @patch("backtest_api.walk_forward.subprocess.run")
    def test_subprocess_called_with_correct_args(self, mock_subprocess, monkeypatch):
        """Verify subprocess command format and env-driven mounts."""
        monkeypatch.setenv("TRADING_STRATEGIES_DIR", "/data/strategies")
        monkeypatch.setenv("TRADING_FT_DATA_DIR", "/data/ftdata")
        mock_subprocess.return_value = MagicMock(
            returncode=0, stderr="", stdout=""
        )

        run_freqtrade_backtest("MyStrat", "20230101-20240601")

        call_args = mock_subprocess.call_args[0][0]
        assert call_args[0] == "podman"
        assert call_args[1] == "run"
        assert "--rm" in call_args
        assert "--userns=keep-id:uid=1000,gid=1000" in call_args
        assert "docker.io/freqtradeorg/freqtrade:stable" in call_args
        assert "backtesting" in call_args
        assert "MyStrat" in call_args
        assert call_args[call_args.index("--strategy-path") + 1] == "/freqtrade/strategies"
        assert "/data/strategies:/freqtrade/strategies:ro,Z" in call_args
        assert "/data/ftdata:/freqtrade/user_data/data:ro,Z" in call_args
        assert "20230101-20240601" in call_args


def _write_feathers(data_dir: Path, first: str, last: str) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range(first, last, freq="4h", tz="UTC")
    for i, pair in enumerate(PAIRS):
        close = 100.0 * (1 + i) * np.exp(np.cumsum(np.sin(np.arange(len(ts)) / 7) * 0.01))
        pd.DataFrame({"timestamp": ts, "open": close, "high": close, "low": close,
                      "close": close, "volume": 1.0}).to_feather(data_dir / f"{pair}.feather")


@pytest.fixture
def box_data(tmp_path, monkeypatch):
    """Feathers shaped like the box's on 2026-10-05: 2023-12-16 04:00 to 2026-10-05 16:00."""
    monkeypatch.setenv("TRADING_DATA_DIR", str(tmp_path / "data"))
    _write_feathers(tmp_path / "data", "2023-12-16 04:00", "2026-10-05 16:00")
    return tmp_path / "data"


def _ft_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _result(start, end, *, trades=20, pf=1.5, dd=0.10, profit=0.05, shift=timedelta(0)):
    """A Freqtrade result for the window [start, end) (end exclusive)."""
    return {"strategy": {"S": {
        "total_trades": trades, "profit_factor": pf, "max_drawdown_account": dd,
        "profit_total": profit,
        "backtest_start": _ft_time(start + shift), "backtest_end": _ft_time(end - CANDLE),
    }}}


HOLDOUT = datetime(2026, 4, 6, tzinfo=timezone.utc)


class TestDefaultWindows:
    """Three equal, consecutive periods inside the development data."""

    def test_common_candle_range_reads_feathers(self, box_data):
        first, last = common_candle_range()
        assert first == datetime(2023, 12, 16, 4, tzinfo=timezone.utc)
        assert last == datetime(2026, 10, 5, 16, tzinfo=timezone.utc)

    def test_periods_are_equal_consecutive_and_labelled(self, box_data):
        windows = default_windows()
        assert [w[2] for w in windows] == ["period 1", "period 2", "period 3"]
        assert windows[0][1] == windows[1][0]
        assert windows[1][1] == windows[2][0]
        lengths = {end - start for start, end, _ in windows}
        assert len(lengths) == 1
        assert lengths.pop() % CANDLE == timedelta(0)

    def test_period_1_leaves_room_for_warmup(self, box_data):
        first, _ = common_candle_range()
        start = default_windows()[0][0]
        assert start >= first + WARMUP
        assert start - (first + WARMUP) < 3 * CANDLE  # only the remainder is dropped

    def test_no_period_touches_the_holdout(self, box_data):
        windows = default_windows()
        assert windows[-1][1] == HOLDOUT
        for start, end, _ in windows:
            last_candle = end - CANDLE
            assert last_candle < HOLDOUT
            # Freqtrade keeps candles with open time <= the timerange stop.
            stop = int(ft_timerange(start, last_candle).split("-")[1])
            assert stop < HOLDOUT.timestamp()

    def test_box_dates(self, box_data):
        """The periods produced from the box's data as of 2026-10-05."""
        assert [(s, e) for s, e, _ in default_windows()] == [
            (datetime(2024, 1, 5, 12, tzinfo=timezone.utc), datetime(2024, 10, 5, 8, tzinfo=timezone.utc)),
            (datetime(2024, 10, 5, 8, tzinfo=timezone.utc), datetime(2025, 7, 6, 4, tzinfo=timezone.utc)),
            (datetime(2025, 7, 6, 4, tzinfo=timezone.utc), HOLDOUT),
        ]


class TestWalkForwardWindow:
    """Phase 2 gates per period."""

    @pytest.fixture(autouse=True)
    def _windows(self, box_data):
        self.windows = default_windows()

    def _run(self, mock_backtest, per_window=None):
        """per_window: kwargs for _result keyed by period index."""
        per_window = per_window or {}
        mock_backtest.side_effect = [
            _result(s, e, **per_window.get(i, {})) for i, (s, e, _) in enumerate(self.windows)
        ]
        return walk_forward_test("TestStrat")

    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_all_windows_pass(self, mock_backtest):
        outcome = self._run(mock_backtest)
        assert outcome["passed"] is True
        assert len(outcome["windows"]) == 3
        assert all(w["passed"] for w in outcome["windows"])

    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_reports_stats_and_actual_range(self, mock_backtest):
        w = self._run(mock_backtest, {0: {"trades": 12, "pf": 1.33333, "dd": 0.15678, "profit": 0.0712}})["windows"][0]
        start, end, _ = self.windows[0]
        assert w["label"] == "period 1"
        assert w["timerange"] == ft_timerange(start, end - CANDLE)
        assert (w["trades"], w["profit_total"], w["profit_factor"], w["max_drawdown"]) == (12, 0.0712, 1.3333, -0.1568)
        assert w["backtest_start"] == _ft_time(start)
        assert w["backtest_end"] == _ft_time(end - CANDLE)
        assert set(w["gate"]) == {"total_trades", "profit_factor", "max_drawdown", "profit_total"}

    @pytest.mark.parametrize("kwargs, check", [
        ({"pf": MIN_PROFIT_FACTOR - 0.01}, "profit_factor"),
        ({"dd": -MAX_DRAWDOWN + 0.01}, "max_drawdown"),
        ({"trades": MIN_TRADES - 1}, "total_trades"),
        ({"profit": 0.0}, "profit_total"),
        ({"profit": -0.01}, "profit_total"),
    ])
    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_each_gate_fails_the_window(self, mock_backtest, kwargs, check):
        outcome = self._run(mock_backtest, {1: kwargs})
        assert outcome["passed"] is False
        assert outcome["windows"][1]["passed"] is False
        assert outcome["windows"][1]["gate"][check]["passed"] is False
        assert outcome["windows"][0]["passed"] is True

    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_boundaries_pass(self, mock_backtest):
        outcome = self._run(mock_backtest, {0: {"pf": MIN_PROFIT_FACTOR, "dd": -MAX_DRAWDOWN, "trades": MIN_TRADES}})
        assert outcome["windows"][0]["passed"] is True

    @pytest.mark.parametrize("pf", [0, None])
    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_no_losing_trades_passes_profit_factor(self, mock_backtest, pf):
        w = self._run(mock_backtest, {0: {"pf": pf}})["windows"][0]
        assert w["passed"] is True
        assert w["gate"]["profit_factor"] == {"value": None, "threshold": MIN_PROFIT_FACTOR, "passed": True}

    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_missing_profit_factor_with_too_few_trades_fails(self, mock_backtest):
        w = self._run(mock_backtest, {0: {"pf": 0, "trades": 5}})["windows"][0]
        assert w["gate"]["profit_factor"]["passed"] is False

    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_start_gap_fails_window(self, mock_backtest):
        w = self._run(mock_backtest, {0: {"shift": timedelta(days=2)}})["windows"][0]
        assert w["passed"] is False
        assert "wrong range" in w["error"] and "started" in w["error"]

    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_one_candle_off_is_tolerated(self, mock_backtest):
        assert self._run(mock_backtest, {0: {"shift": CANDLE}})["windows"][0]["passed"] is True

    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_end_gap_fails_window(self, mock_backtest):
        results = [_result(s, e) for s, e, _ in self.windows]
        results[2]["strategy"]["S"]["backtest_end"] = "2026-04-02 20:00:00"
        mock_backtest.side_effect = results
        w = walk_forward_test("TestStrat")["windows"][2]
        assert w["passed"] is False
        assert "ended 2026-04-02 20:00" in w["error"]

    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_window_with_error(self, mock_backtest):
        s, e, _ = self.windows[0]
        mock_backtest.side_effect = [_result(s, e), {"error": "Insufficient data"}, _result(s, e)]
        outcome = walk_forward_test("TestStrat")
        assert outcome["passed"] is False
        assert outcome["windows"][1]["error"] == "Insufficient data"

    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_window_returns_none(self, mock_backtest):
        s, e, _ = self.windows[0]
        mock_backtest.side_effect = [_result(s, e), None, _result(s, e)]
        outcome = walk_forward_test("TestStrat")
        assert outcome["passed"] is False
        assert outcome["windows"][1]["error"] == "No results"
