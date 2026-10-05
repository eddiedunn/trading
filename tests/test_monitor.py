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
    best_candidate,
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

    def test_low_win_rate_still_passes(self):
        """Win rate is not gated: a 30% win rate with big winners passes."""
        metrics = {
            "trade_count": 30,
            "profit_pct": 10.0,
            "max_drawdown": -10.0,
            "win_rate": 0.30,
            "profit_factor": 1.5,
        }
        assert meets_promotion_criteria(metrics) is True

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
                "max_drawdown": 0.12,
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
    def test_collect_metrics_null_profit_factor(self, mock_client_cls):
        """A fresh instance reports profit_factor null; treat it as 0, not a crash."""
        instance = PaperInstance("S", 8090, "paper_s_0", "paper_s")
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.side_effect = [
            MagicMock(json=lambda: {"profit_all_percent": 0.0, "trade_count": 2,
                                    "winrate": 0.0, "profit_factor": None, "max_drawdown": 0.0}),
            MagicMock(json=lambda: []),
        ]

        metrics = collect_metrics(instance)

        assert metrics["profit_factor"] == 0
        assert metrics["max_drawdown"] == 0
        assert meets_promotion_criteria(metrics) is False

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


def _m(strategy, **kw):
    base = {"strategy": strategy, "trade_count": 30, "profit_pct": 10.0,
            "max_drawdown": -10.0, "win_rate": 0.50, "profit_factor": 1.5}
    return {**base, **kw}


FAILING = {"trade_count": 5, "profit_pct": 2.0, "max_drawdown": -20.0, "profit_factor": 1.1}


class TestRunPaperArena:
    """Test the main evaluation loop. One time.time() call sets the deadline,
    then one per loop check; after the loop comes one final poll."""

    S1 = PaperInstance("Strat1", 8090, "paper_strat1_0", "paper_strat1")
    S2 = PaperInstance("Strat2", 8091, "paper_strat2_1", "paper_strat2")

    @patch("paper.monitor.time.time")
    @patch("paper.monitor.time.sleep")
    @patch("paper.monitor.collect_metrics")
    def test_promotes_best_of_final_poll(self, mock_collect, mock_sleep, mock_time, capsys):
        mock_time.side_effect = [0, 0, 100]
        mock_collect.side_effect = [
            _m("Strat1", profit_factor=1.3), _m("Strat2"),  # in-loop poll
            _m("Strat1", profit_factor=1.3), _m("Strat2"),  # final poll
        ]

        final = run_paper_arena([self.S1, self.S2], eval_days=0.00001)

        assert [m["strategy"] for m in final] == ["Strat1", "Strat2"]
        assert "PROMOTION CANDIDATE: Strat2" in capsys.readouterr().out

    @patch("paper.monitor.time.time")
    @patch("paper.monitor.time.sleep")
    @patch("paper.monitor.collect_metrics")
    def test_no_candidates(self, mock_collect, mock_sleep, mock_time, capsys):
        mock_time.side_effect = [0, 0, 100]
        mock_collect.side_effect = [_m("Strat1", **FAILING), _m("Strat1", **FAILING)]

        final = run_paper_arena([self.S1], eval_days=0.00001)

        assert best_candidate(final) is None
        assert "PROMOTION" not in capsys.readouterr().out

    @patch("paper.monitor.time.time")
    @patch("paper.monitor.time.sleep")
    @patch("paper.monitor.collect_metrics")
    def test_passes_mid_run_but_fails_at_end_is_not_promoted(self, mock_collect, mock_sleep, mock_time, capsys):
        """A strategy that met the bar mid-window and fell below it is not promoted."""
        mock_time.side_effect = [0, 0, 1500, 3000]
        mock_collect.side_effect = [
            _m("Strat1"),                 # poll 1: passes
            _m("Strat1", profit_pct=1.0),  # poll 2: below the bar
            _m("Strat1", profit_pct=1.0),  # final poll: still below
        ]

        final = run_paper_arena([self.S1], eval_days=2500 / 86400)

        assert best_candidate(final) is None
        assert meets_promotion_criteria(final[0]) is False
        out = capsys.readouterr().out
        assert "PROMOTION CANDIDATE" not in out
        assert "promote --strategy" not in out

    @patch("paper.monitor.time.time")
    @patch("paper.monitor.time.sleep")
    @patch("paper.monitor.collect_metrics")
    @patch("paper.monitor._write_metrics_snapshot")
    def test_writes_every_poll_to_db(self, mock_write, mock_collect, mock_sleep, mock_time):
        mock_time.side_effect = [0, 0, 100]
        mock_collect.side_effect = [_m("Strat1"), _m("Strat1")]

        run_paper_arena([self.S1], eval_days=0.00001, db_url="postgresql://localhost/trading")

        assert mock_write.call_count == 2

    @patch("paper.monitor.time.time")
    @patch("paper.monitor.time.sleep")
    @patch("paper.monitor.collect_metrics")
    def test_handles_collection_error(self, mock_collect, mock_sleep, mock_time):
        mock_time.side_effect = [0, 0, 100]
        mock_collect.side_effect = Exception("Connection failed")

        assert run_paper_arena([self.S1], eval_days=0.00001) == []

    @patch("paper.monitor.time.time")
    @patch("paper.monitor.time.sleep")
    @patch("paper.monitor.collect_metrics")
    def test_respects_deadline(self, mock_collect, mock_sleep, mock_time):
        """Deadline 2500s: loop polls at t=0 and t=1500, stops at t=3000, then polls once more."""
        mock_time.side_effect = [0, 0, 1500, 3000]
        mock_collect.side_effect = [_m("Strat1", **FAILING)] * 3

        run_paper_arena([self.S1], eval_days=2500 / 86400)

        assert mock_collect.call_count == 3


class TestQueue:
    """Test the strategy_registry queue the service loop reads."""

    @patch("paper.monitor.psycopg2.connect")
    def test_queued_strategies_selects_unstarted_oldest_first(self, mock_connect):
        from paper.monitor import queued_strategies

        cur = mock_connect.return_value.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = [("EmaCross",), ("Other",)]

        assert queued_strategies("postgresql://x", limit=6) == ["EmaCross", "Other"]
        sql, params = cur.execute.call_args[0]
        assert "paper_queued_at IS NOT NULL" in sql
        assert "paper_started_at IS NULL" in sql
        assert "ORDER BY paper_queued_at" in sql
        assert params == (6,)

    @patch("paper.monitor.psycopg2.connect")
    def test_requeue_interrupted_clears_unfinished_starts(self, mock_connect):
        from paper.monitor import requeue_interrupted

        cur = mock_connect.return_value.cursor.return_value.__enter__.return_value
        cur.description = None

        requeue_interrupted("postgresql://x")

        sql = cur.execute.call_args[0][0]
        assert "SET paper_started_at = NULL" in sql
        assert "paper_finished_at IS NULL" in sql

    @patch.dict(os.environ, {"POSTGRES_USER": "u", "POSTGRES_PASSWORD": "p", "POSTGRES_DB": "d"})
    def test_db_url_defaults_to_local_host(self):
        from paper.monitor import db_url_from_env

        assert db_url_from_env() == "postgresql://u:p@127.0.0.1:5432/d"


class TestRunCohort:
    """Test one paper cohort from spawn to recorded outcome."""

    @patch("paper.monitor.teardown_paper_instance")
    @patch("paper.monitor._record_result")
    @patch("paper.monitor._mark_started")
    @patch("paper.monitor.run_paper_arena")
    @patch("paper.monitor.wait_until_ready", return_value=True)
    @patch("paper.monitor.spawn_paper_instance")
    def test_records_each_outcome_and_tears_down(
        self, mock_spawn, mock_ready, mock_arena, mock_started, mock_record, mock_teardown
    ):
        from paper.monitor import run_cohort

        a = PaperInstance("Good", 8090, "paper_good_0", "paper_good")
        b = PaperInstance("Bad", 8091, "paper_bad_1", "paper_bad")
        c = PaperInstance("Gone", 8092, "paper_gone_2", "paper_gone")
        mock_spawn.side_effect = [a, b, c]
        # "Gone" answered no final poll, so it has no metrics.
        mock_arena.return_value = [_m("Good"), _m("Bad", profit_factor=0.5)]

        run_cohort("postgresql://x", ["Good", "Bad", "Gone"], eval_days=14)

        assert mock_spawn.call_args_list == [call("Good", 0), call("Bad", 1), call("Gone", 2)]
        mock_started.assert_called_once_with("postgresql://x", ["Good", "Bad", "Gone"])
        mock_arena.assert_called_once_with([a, b, c], eval_days=14, db_url="postgresql://x")
        assert mock_record.call_args_list == [
            call("postgresql://x", "Good", True),
            call("postgresql://x", "Bad", False),
            call("postgresql://x", "Gone", False),
        ]
        assert mock_teardown.call_count == 3

    @patch("paper.monitor.teardown_paper_instance")
    @patch("paper.monitor._record_result")
    @patch("paper.monitor._mark_started")
    @patch("paper.monitor._write_metrics_snapshot")
    @patch("paper.monitor.collect_metrics")
    @patch("paper.monitor.time.sleep")
    @patch("paper.monitor.time.time")
    @patch("paper.monitor.wait_until_ready", return_value=True)
    @patch("paper.monitor.spawn_paper_instance")
    def test_mid_run_pass_then_end_fail_records_fail_and_prints_no_promotion(
        self, mock_spawn, mock_ready, mock_time, mock_sleep, mock_collect, mock_write,
        mock_started, mock_record, mock_teardown, capsys,
    ):
        """Printed line and recorded result both come from the final poll."""
        from paper.monitor import run_cohort

        mock_spawn.return_value = PaperInstance("Fader", 8090, "paper_fader_0", "paper_fader")
        mock_time.side_effect = [0, 0, 100]
        mock_collect.side_effect = [_m("Fader"), _m("Fader", **FAILING)]

        run_cohort("postgresql://x", ["Fader"], eval_days=0.00001)

        mock_record.assert_called_once_with("postgresql://x", "Fader", False)
        assert "PROMOTION CANDIDATE" not in capsys.readouterr().out

    @patch("paper.monitor.teardown_paper_instance")
    @patch("paper.monitor._mark_started")
    @patch("paper.monitor.run_paper_arena", side_effect=RuntimeError("boom"))
    @patch("paper.monitor.wait_until_ready", return_value=True)
    @patch("paper.monitor.spawn_paper_instance")
    def test_tears_down_when_arena_fails(self, mock_spawn, mock_ready, mock_arena, mock_started, mock_teardown):
        from paper.monitor import run_cohort

        mock_spawn.return_value = PaperInstance("S", 8090, "paper_s_0", "paper_s")

        with pytest.raises(RuntimeError):
            run_cohort("postgresql://x", ["S"])

        mock_teardown.assert_called_once()
