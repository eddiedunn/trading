"""Unit tests for paper arena monitor.

Tests promotion criteria evaluation, metric extraction from REST API,
and Postgres snapshot persistence.
"""

import os
from unittest.mock import patch, MagicMock, call

import pytest

from paper.monitor import (
    collect_metrics,
    meets_promotion_criteria,
    run_paper_arena,
    _write_metrics_snapshot,
    PROMOTION_CRITERIA,
)
from paper.orchestrator import PaperInstance


class TestMeetsPromotionCriteria:
    """Test the promotion gate function."""

    def test_all_criteria_pass(self):
        """Metrics meeting all criteria return True."""
        metrics = {
            "trade_count": 30,
            "profit_pct": 10.0,
            "max_drawdown": -10.0,
            "win_rate": 0.50,
            "profit_factor": 1.5,
        }
        assert meets_promotion_criteria(metrics) is True

    def test_trade_count_boundary(self):
        """Test trade_count at boundary."""
        metrics = {
            "trade_count": PROMOTION_CRITERIA["min_trades"],
            "profit_pct": 10.0,
            "max_drawdown": -10.0,
            "win_rate": 0.50,
            "profit_factor": 1.5,
        }
        assert meets_promotion_criteria(metrics) is True

        metrics["trade_count"] = PROMOTION_CRITERIA["min_trades"] - 1
        assert meets_promotion_criteria(metrics) is False

    def test_profit_pct_boundary(self):
        """Test profit_pct at boundary."""
        metrics = {
            "trade_count": 30,
            "profit_pct": PROMOTION_CRITERIA["min_profit_pct"],
            "max_drawdown": -10.0,
            "win_rate": 0.50,
            "profit_factor": 1.5,
        }
        assert meets_promotion_criteria(metrics) is True

        metrics["profit_pct"] = PROMOTION_CRITERIA["min_profit_pct"] - 0.1
        assert meets_promotion_criteria(metrics) is False

    def test_max_drawdown_boundary(self):
        """Test max_drawdown at boundary (remember: more negative is worse)."""
        metrics = {
            "trade_count": 30,
            "profit_pct": 10.0,
            "max_drawdown": PROMOTION_CRITERIA["max_drawdown_pct"],
            "win_rate": 0.50,
            "profit_factor": 1.5,
        }
        assert meets_promotion_criteria(metrics) is True

        # More negative (worse) → fail
        metrics["max_drawdown"] = PROMOTION_CRITERIA["max_drawdown_pct"] - 5.0
        assert meets_promotion_criteria(metrics) is False

    def test_win_rate_boundary(self):
        """Test win_rate at boundary."""
        metrics = {
            "trade_count": 30,
            "profit_pct": 10.0,
            "max_drawdown": -10.0,
            "win_rate": PROMOTION_CRITERIA["min_win_rate"],
            "profit_factor": 1.5,
        }
        assert meets_promotion_criteria(metrics) is True

        metrics["win_rate"] = PROMOTION_CRITERIA["min_win_rate"] - 0.01
        assert meets_promotion_criteria(metrics) is False

    def test_profit_factor_boundary(self):
        """Test profit_factor at boundary."""
        metrics = {
            "trade_count": 30,
            "profit_pct": 10.0,
            "max_drawdown": -10.0,
            "win_rate": 0.50,
            "profit_factor": PROMOTION_CRITERIA["min_profit_factor"],
        }
        assert meets_promotion_criteria(metrics) is True

        metrics["profit_factor"] = PROMOTION_CRITERIA["min_profit_factor"] - 0.01
        assert meets_promotion_criteria(metrics) is False

    def test_fails_multiple_criteria(self):
        """Failing multiple criteria still returns False."""
        metrics = {
            "trade_count": 5,  # too low
            "profit_pct": 1.0,  # too low
            "max_drawdown": -20.0,
            "win_rate": 0.50,
            "profit_factor": 1.5,
        }
        assert meets_promotion_criteria(metrics) is False


class TestCollectMetrics:
    """Test metrics collection from Freqtrade REST API."""

    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
    @patch("paper.monitor.httpx.Client")
    def test_collect_metrics_success(self, mock_client_cls):
        """Collect metrics from running instance."""
        instance = PaperInstance(
            strategy_name="TestStrat",
            port=8090,
            container_name="paper_teststrat_0",
            db_schema="paper_teststrat",
        )

        # Mock httpx client
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.side_effect = [
            MagicMock(json=lambda: {
                "profit_all_percent": 8.5,
                "trade_count": 25,
                "winrate": 0.48,
                "profit_factor": 1.3,
                "max_drawdown": -12.0,
            }),
            MagicMock(json=lambda: [
                {"pair": "BTC/USDC:USDC"},
                {"pair": "SOL/USDC:USDC"},
            ]),
        ]

        metrics = collect_metrics(instance)

        assert metrics["strategy"] == "TestStrat"
        assert metrics["profit_pct"] == 8.5
        assert metrics["trade_count"] == 25
        assert metrics["win_rate"] == 0.48
        assert metrics["profit_factor"] == 1.3
        assert metrics["max_drawdown"] == -12.0
        assert metrics["open_trades"] == 2

    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
    @patch("paper.monitor.httpx.Client")
    def test_collect_metrics_status_not_list(self, mock_client_cls):
        """Handle case where status is not a list."""
        instance = PaperInstance(
            strategy_name="TestStrat",
            port=8090,
            container_name="paper_teststrat_0",
            db_schema="paper_teststrat",
        )

        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.side_effect = [
            MagicMock(json=lambda: {
                "profit_all_percent": 8.5,
                "trade_count": 25,
                "winrate": 0.48,
                "profit_factor": 1.3,
                "max_drawdown": -12.0,
            }),
            MagicMock(json=lambda: {}),  # not a list
        ]

        metrics = collect_metrics(instance)

        assert metrics["open_trades"] == 0

    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
    @patch("paper.monitor.httpx.Client")
    def test_collect_metrics_uses_auth(self, mock_client_cls):
        """Verify authentication is set."""
        instance = PaperInstance(
            strategy_name="TestStrat",
            port=8090,
            container_name="paper_teststrat_0",
            db_schema="paper_teststrat",
        )

        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.side_effect = [
            MagicMock(json=lambda: {}),
            MagicMock(json=lambda: []),
        ]

        collect_metrics(instance)

        # Check Client was called with auth
        call_kwargs = mock_client_cls.call_args[1]
        assert call_kwargs["auth"] == ("freqtrade", "changeme")

    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
    @patch("paper.monitor.httpx.Client")
    def test_collect_metrics_calls_correct_endpoints(self, mock_client_cls):
        """Verify correct API endpoints are called."""
        instance = PaperInstance(
            strategy_name="TestStrat",
            port=8090,
            container_name="paper_teststrat_0",
            db_schema="paper_teststrat",
        )

        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.side_effect = [
            MagicMock(json=lambda: {}),
            MagicMock(json=lambda: []),
        ]

        collect_metrics(instance)

        calls = mock_client.get.call_args_list
        assert len(calls) == 2
        assert "8090" in calls[0][0][0]
        assert "/api/v1/profit" in calls[0][0][0]
        assert "/api/v1/status" in calls[1][0][0]


class TestWriteMetricsSnapshot:
    """Test persistence of metrics to Postgres."""

    @patch("paper.monitor.psycopg2.connect")
    def test_write_metrics_snapshot(self, mock_connect):
        """Write metrics to paper_snapshots table."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        metrics = [
            {
                "strategy": "TestStrat1",
                "profit_pct": 8.5,
                "trade_count": 25,
                "win_rate": 0.48,
                "profit_factor": 1.3,
                "max_drawdown": -12.0,
            },
            {
                "strategy": "TestStrat2",
                "profit_pct": 10.0,
                "trade_count": 30,
                "win_rate": 0.50,
                "profit_factor": 1.5,
                "max_drawdown": -10.0,
            },
        ]

        _write_metrics_snapshot(metrics, "postgresql://localhost/trading")

        # Verify INSERT was called twice
        assert mock_cursor.execute.call_count == 2

        # Verify commit was called
        mock_conn.commit.assert_called_once()

    @patch("paper.monitor.psycopg2.connect")
    def test_write_metrics_snapshot_sql_shape(self, mock_connect):
        """Verify SQL INSERT statement shape."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        metrics = [
            {
                "strategy": "TestStrat",
                "profit_pct": 8.5,
                "trade_count": 25,
                "win_rate": 0.48,
                "profit_factor": 1.3,
                "max_drawdown": -12.0,
            }
        ]

        _write_metrics_snapshot(metrics, "postgresql://localhost/trading")

        # Check the SQL statement
        call_args = mock_cursor.execute.call_args
        sql = call_args[0][0]
        assert "INSERT INTO paper_snapshots" in sql
        assert "ts" in sql
        assert "strategy" in sql
        assert "profit_pct" in sql
        assert "trade_count" in sql
        assert "win_rate" in sql
        assert "profit_factor" in sql
        assert "max_drawdown" in sql

    @patch("paper.monitor.psycopg2.connect")
    def test_write_metrics_snapshot_closes_connection(self, mock_connect):
        """Verify connection is closed after write."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        metrics = [{"strategy": "Test", "profit_pct": 5.0, "trade_count": 20,
                    "win_rate": 0.45, "profit_factor": 1.2, "max_drawdown": -15.0}]

        _write_metrics_snapshot(metrics, "postgresql://localhost/trading")

        # Verify close was called
        mock_conn.close.assert_called_once()

    @patch("paper.monitor.psycopg2.connect")
    def test_write_metrics_snapshot_closes_on_error(self, mock_connect):
        """Verify connection is closed even if error occurs."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.execute.side_effect = Exception("DB error")

        metrics = [{"strategy": "Test", "profit_pct": 5.0, "trade_count": 20,
                    "win_rate": 0.45, "profit_factor": 1.2, "max_drawdown": -15.0}]

        with pytest.raises(Exception):
            _write_metrics_snapshot(metrics, "postgresql://localhost/trading")

        # Close should still be called
        mock_conn.close.assert_called_once()


class TestRunPaperArena:
    """Test the main evaluation loop."""

    @patch("paper.monitor.time.time")
    @patch("paper.monitor.time.sleep")
    @patch("paper.monitor.collect_metrics")
    def test_run_paper_arena_finds_best(self, mock_collect, mock_sleep, mock_time):
        """Find best performer meeting criteria."""
        instance1 = PaperInstance(
            strategy_name="Strat1",
            port=8090,
            container_name="paper_strat1_0",
            db_schema="paper_strat1",
        )
        instance2 = PaperInstance(
            strategy_name="Strat2",
            port=8091,
            container_name="paper_strat2_1",
            db_schema="paper_strat2",
        )

        # Time sequence: initial deadline calc, loop condition check (true), loop condition check (false)
        mock_time.side_effect = [0, 0, 100]

        # Both instances pass criteria on first poll
        mock_collect.side_effect = [
            {
                "strategy": "Strat1",
                "trade_count": 25,
                "profit_pct": 8.0,
                "max_drawdown": -12.0,
                "win_rate": 0.48,
                "profit_factor": 1.3,
            },
            {
                "strategy": "Strat2",
                "trade_count": 30,
                "profit_pct": 10.0,
                "max_drawdown": -10.0,
                "win_rate": 0.50,
                "profit_factor": 1.5,
            },
        ]

        best = run_paper_arena([instance1, instance2], eval_days=0.00001)

        assert best is not None
        assert best["strategy"] == "Strat2"
        assert best["profit_factor"] == 1.5

    @patch("paper.monitor.time.time")
    @patch("paper.monitor.time.sleep")
    @patch("paper.monitor.collect_metrics")
    def test_run_paper_arena_no_candidates(self, mock_collect, mock_sleep, mock_time):
        """Return None if no instance meets criteria."""
        instance = PaperInstance(
            strategy_name="Strat",
            port=8090,
            container_name="paper_strat_0",
            db_schema="paper_strat",
        )

        # Time sequence: initial calc, loop check (true), loop check (false)
        mock_time.side_effect = [0, 0, 100]

        # Instance fails criteria
        mock_collect.side_effect = [
            {
                "strategy": "Strat",
                "trade_count": 5,  # too low
                "profit_pct": 2.0,  # too low
                "max_drawdown": -20.0,
                "win_rate": 0.40,
                "profit_factor": 1.1,
            },
        ]

        best = run_paper_arena([instance], eval_days=0.00001)

        assert best is None

    @patch("paper.monitor.time.time")
    @patch("paper.monitor.time.sleep")
    @patch("paper.monitor.collect_metrics")
    @patch("paper.monitor._write_metrics_snapshot")
    def test_run_paper_arena_writes_metrics_to_db(self, mock_write, mock_collect, mock_sleep, mock_time):
        """Write metrics to DB if db_url provided."""
        instance = PaperInstance(
            strategy_name="Strat",
            port=8090,
            container_name="paper_strat_0",
            db_schema="paper_strat",
        )

        mock_time.side_effect = [0, 0, 100]
        mock_collect.side_effect = [
            {
                "strategy": "Strat",
                "trade_count": 25,
                "profit_pct": 8.0,
                "max_drawdown": -12.0,
                "win_rate": 0.48,
                "profit_factor": 1.3,
            },
        ]

        run_paper_arena([instance], eval_days=0.00001, db_url="postgresql://localhost/trading")

        # Verify snapshot write was called
        mock_write.assert_called()

    @patch("paper.monitor.time.time")
    @patch("paper.monitor.time.sleep")
    @patch("paper.monitor.collect_metrics")
    def test_run_paper_arena_handles_collection_error(self, mock_collect, mock_sleep, mock_time):
        """Handle errors during metric collection gracefully."""
        instance = PaperInstance(
            strategy_name="Strat",
            port=8090,
            container_name="paper_strat_0",
            db_schema="paper_strat",
        )

        mock_time.side_effect = [0, 0, 100]
        mock_collect.side_effect = Exception("Connection failed")

        # Should not raise, returns None
        best = run_paper_arena([instance], eval_days=0.00001)

        assert best is None

    @patch("paper.monitor.time.time")
    @patch("paper.monitor.time.sleep")
    @patch("paper.monitor.collect_metrics")
    def test_run_paper_arena_respects_deadline(self, mock_collect, mock_sleep, mock_time):
        """Loop stops when deadline passed."""
        instance = PaperInstance(
            strategy_name="Strat",
            port=8090,
            container_name="paper_strat_0",
            db_schema="paper_strat",
        )

        # Time sequence: initial calc=0, check1=0 (0 < 2500=deadline), check2=1500 (1500 < 2500), check3=3000 (3000 < 2500 is False)
        # With eval_days = 2500/86400 ≈ 0.0289, deadline = 0 + 2500/86400*86400 = 2500
        mock_time.side_effect = [0, 0, 1500, 3000]
        mock_collect.side_effect = [
            {"strategy": "Strat", "trade_count": 10, "profit_pct": 2.0,
             "max_drawdown": -20.0, "win_rate": 0.40, "profit_factor": 1.1},
            {"strategy": "Strat", "trade_count": 20, "profit_pct": 5.0,
             "max_drawdown": -15.0, "win_rate": 0.45, "profit_factor": 1.2},
        ]

        run_paper_arena([instance], eval_days=2500/86400)

        # Should have looped twice, then exited on third time check
        assert mock_collect.call_count == 2
