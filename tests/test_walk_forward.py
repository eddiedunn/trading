"""Unit tests for Phase 2 walk-forward validation.

Tests gate logic for profit_factor and max_drawdown per window,
result aggregation across 3 windows, and Freqtrade result parsing.
"""

import json
import zipfile
from datetime import date
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from backtest_api.walk_forward import (
    run_freqtrade_backtest,
    walk_forward_test,
    _extract_profit_factor,
    _extract_max_drawdown,
    MIN_PROFIT_FACTOR,
    MAX_DRAWDOWN,
    default_windows,
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


class TestDefaultWindows:
    """Windows are anchored to today because Hyperliquid history is short."""

    def test_windows_are_contiguous_and_end_today(self):
        windows = default_windows(date(2026, 10, 2))
        assert [w[2] for w in windows] == ["in-sample", "validation", "out-of-sample"]
        assert windows[-1][1] == "20261002"
        assert windows[0][1] == windows[1][0]
        assert windows[1][1] == windows[2][0]

    def test_windows_stay_inside_kept_history(self):
        """History is kept from 2023-12; the in-sample window must not start before it."""
        assert default_windows(date(2026, 10, 2))[0][0] >= "20231216"


class TestWalkForwardWindow:
    """Test walk-forward across 3 windows with gate logic."""

    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_all_windows_pass(self, mock_backtest):
        """All windows pass → overall pass."""
        # Mock each window's results
        results = [
            {"strategy": {"S": {"profit_factor": 1.3, "max_drawdown": -0.15}}},
            {"strategy": {"S": {"profit_factor": 1.25, "max_drawdown": -0.18}}},
            {"strategy": {"S": {"profit_factor": 1.5, "max_drawdown": -0.12}}},
        ]
        mock_backtest.side_effect = results

        outcome = walk_forward_test("TestStrat")

        assert outcome["passed"] is True
        assert len(outcome["windows"]) == 3
        assert all(w["passed"] for w in outcome["windows"])

    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_one_window_fails_profit_factor(self, mock_backtest):
        """One window fails PF gate → overall fail."""
        results = [
            {"strategy": {"S": {"profit_factor": 1.3, "max_drawdown": -0.15}}},  # pass
            {"strategy": {"S": {"profit_factor": 1.0, "max_drawdown": -0.18}}},  # fail PF < 1.2
            {"strategy": {"S": {"profit_factor": 1.5, "max_drawdown": -0.12}}},  # pass
        ]
        mock_backtest.side_effect = results

        outcome = walk_forward_test("TestStrat")

        assert outcome["passed"] is False
        assert outcome["windows"][1]["passed"] is False
        assert outcome["windows"][0]["passed"] is True
        assert outcome["windows"][2]["passed"] is True

    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_one_window_fails_drawdown(self, mock_backtest):
        """One window fails DD gate → overall fail."""
        results = [
            {"strategy": {"S": {"profit_factor": 1.3, "max_drawdown": -0.15}}},
            {"strategy": {"S": {"profit_factor": 1.25, "max_drawdown": -0.30}}},  # fail DD < -0.25
            {"strategy": {"S": {"profit_factor": 1.5, "max_drawdown": -0.12}}},
        ]
        mock_backtest.side_effect = results

        outcome = walk_forward_test("TestStrat")

        assert outcome["passed"] is False
        assert outcome["windows"][1]["passed"] is False

    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_window_with_error(self, mock_backtest):
        """Window with error → overall fail."""
        results = [
            {"strategy": {"S": {"profit_factor": 1.3, "max_drawdown": -0.15}}},
            {"error": "Insufficient data"},
            {"strategy": {"S": {"profit_factor": 1.5, "max_drawdown": -0.12}}},
        ]
        mock_backtest.side_effect = results

        outcome = walk_forward_test("TestStrat")

        assert outcome["passed"] is False
        assert outcome["windows"][1]["passed"] is False
        assert "error" in outcome["windows"][1]

    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_window_returns_none(self, mock_backtest):
        """Window returns None → overall fail."""
        results = [
            {"strategy": {"S": {"profit_factor": 1.3, "max_drawdown": -0.15}}},
            None,
            {"strategy": {"S": {"profit_factor": 1.5, "max_drawdown": -0.12}}},
        ]
        mock_backtest.side_effect = results

        outcome = walk_forward_test("TestStrat")

        assert outcome["passed"] is False
        assert outcome["windows"][1]["passed"] is False
        assert outcome["windows"][1]["error"] == "No results"

    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_window_labels_and_timeranges(self, mock_backtest):
        """Verify window labels and timeranges match default_windows()."""
        results = [
            {"strategy": {"S": {"profit_factor": 1.3, "max_drawdown": -0.15}}},
            {"strategy": {"S": {"profit_factor": 1.25, "max_drawdown": -0.18}}},
            {"strategy": {"S": {"profit_factor": 1.5, "max_drawdown": -0.12}}},
        ]
        mock_backtest.side_effect = results

        outcome = walk_forward_test("TestStrat")

        for i, (start, end, label) in enumerate(default_windows()):
            assert outcome["windows"][i]["label"] == label
            assert outcome["windows"][i]["timerange"] == f"{start}-{end}"

    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_metrics_rounded(self, mock_backtest):
        """Metrics in output are rounded to 4 decimals."""
        results = [
            {"strategy": {"S": {"profit_factor": 1.33333, "max_drawdown": -0.15678}}},
            {"strategy": {"S": {"profit_factor": 1.25555, "max_drawdown": -0.18901}}},
            {"strategy": {"S": {"profit_factor": 1.5, "max_drawdown": -0.12}}},
        ]
        mock_backtest.side_effect = results

        outcome = walk_forward_test("TestStrat")

        for w in outcome["windows"]:
            if "profit_factor" in w:
                # Verify it's a rounded value (max 4 decimals)
                pf_str = str(w["profit_factor"])
                decimals = len(pf_str.split(".")[-1]) if "." in pf_str else 0
                assert decimals <= 4

    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_boundary_profit_factor(self, mock_backtest):
        """Test PF boundary at exactly MIN_PROFIT_FACTOR."""
        results = [
            {"strategy": {"S": {"profit_factor": MIN_PROFIT_FACTOR, "max_drawdown": -0.15}}},
            {"strategy": {"S": {"profit_factor": MIN_PROFIT_FACTOR - 0.01, "max_drawdown": -0.15}}},
            {"strategy": {"S": {"profit_factor": 1.5, "max_drawdown": -0.12}}},
        ]
        mock_backtest.side_effect = results

        outcome = walk_forward_test("TestStrat")

        assert outcome["windows"][0]["passed"] is True  # exactly at threshold
        assert outcome["windows"][1]["passed"] is False  # just below threshold

    @patch("backtest_api.walk_forward.run_freqtrade_backtest")
    def test_boundary_max_drawdown(self, mock_backtest):
        """Test DD boundary at exactly MAX_DRAWDOWN."""
        results = [
            {"strategy": {"S": {"profit_factor": 1.3, "max_drawdown": MAX_DRAWDOWN}}},
            {"strategy": {"S": {"profit_factor": 1.3, "max_drawdown": MAX_DRAWDOWN - 0.01}}},
            {"strategy": {"S": {"profit_factor": 1.5, "max_drawdown": -0.12}}},
        ]
        mock_backtest.side_effect = results

        outcome = walk_forward_test("TestStrat")

        assert outcome["windows"][0]["passed"] is True  # exactly at threshold
        assert outcome["windows"][1]["passed"] is False  # worse than threshold
