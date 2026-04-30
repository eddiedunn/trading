"""Unit tests for Phase 2 walk-forward validation.

Tests gate logic for profit_factor and max_drawdown per window,
result aggregation across 3 windows, and Freqtrade result parsing.
"""

import json
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
    WINDOWS,
)


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

    @patch("backtest_api.walk_forward.subprocess.run")
    @patch("builtins.open", create=True)
    def test_successful_backtest(self, mock_open, mock_subprocess):
        """Successful backtest run returns parsed JSON."""
        # Mock subprocess result
        mock_subprocess.return_value = MagicMock(
            returncode=0, stderr="", stdout=""
        )

        # Mock file system
        result_json = {
            "strategy": {"TestStrat": {"profit_factor": 1.3, "max_drawdown": -0.18}}
        }
        mock_file = MagicMock()
        mock_file.read_text.return_value = json.dumps(result_json)
        mock_open.return_value.__enter__.return_value = mock_file

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_text", return_value=json.dumps(result_json)):
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

        with patch("pathlib.Path.exists", return_value=False):
            result = run_freqtrade_backtest("TestStrat", "20230101-20240601")

        assert result is None

    @patch("backtest_api.walk_forward.subprocess.run")
    def test_subprocess_called_with_correct_args(self, mock_subprocess):
        """Verify subprocess command format."""
        mock_subprocess.return_value = MagicMock(
            returncode=0, stderr="", stdout=""
        )

        with patch("pathlib.Path.exists", return_value=False):
            run_freqtrade_backtest("MyStrat", "20230101-20240601")

        # Check that podman run was called
        call_args = mock_subprocess.call_args[0][0]
        assert call_args[0] == "podman"
        assert call_args[1] == "run"
        assert "--rm" in call_args
        assert "freqtradeorg/freqtrade:stable" in call_args
        assert "backtesting" in call_args
        assert "--strategy" in call_args
        assert "MyStrat" in call_args
        assert "--timerange" in call_args
        assert "20230101-20240601" in call_args


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
        """Verify window labels and timeranges match WINDOWS constant."""
        results = [
            {"strategy": {"S": {"profit_factor": 1.3, "max_drawdown": -0.15}}},
            {"strategy": {"S": {"profit_factor": 1.25, "max_drawdown": -0.18}}},
            {"strategy": {"S": {"profit_factor": 1.5, "max_drawdown": -0.12}}},
        ]
        mock_backtest.side_effect = results

        outcome = walk_forward_test("TestStrat")

        for i, (start, end, label) in enumerate(WINDOWS):
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
