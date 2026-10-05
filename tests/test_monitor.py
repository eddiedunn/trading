"""Unit tests for paper arena monitor.

Tests metric collection from the REST API, Postgres snapshots, the queue, and
the run loop that ends each run at 30 closed trades or 60 days.
"""

import os
from unittest.mock import patch, MagicMock, call

import pytest

from paper.monitor import (
    MAX_RUN_DAYS,
    POLL_INTERVAL_SECS,
    TARGET_CLOSED_TRADES,
    collect_metrics,
    collect_trades,
    end_reason,
    finish_run,
    run_paper_arena,
    _write_metrics_snapshot,
)
from paper.orchestrator import PaperInstance

DAY = 86400


def _inst(name="StratA", slot=0):
    return PaperInstance(name, 8090 + slot, f"paper_{name.lower()}_{slot}", f"paper_{name.lower()}")


def _metrics(name="StratA", closed=0, **kw):
    return {"strategy": name, "profit_pct": 1.0, "trade_count": closed, "closed_trade_count": closed,
            "win_rate": 0.5, "profit_factor": 1.2, "max_drawdown": -3.0, "open_trades": 0, **kw}


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
                "closed_trade_count": 23,
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
        assert metrics["closed_trade_count"] == 23
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


class TestCollectTrades:
    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
    @patch("paper.monitor.httpx.Client")
    def test_returns_closed_and_open_trades(self, mock_client_cls):
        client = mock_client_cls.return_value.__enter__.return_value
        client.get.side_effect = [
            MagicMock(json=lambda: {"trades": [{"profit_ratio": -0.02, "profit_abs": -1.0}], "total_trades": 1}),
            MagicMock(json=lambda: [{"profit_ratio": 0.03, "profit_abs": 1.5}]),
        ]

        trades = collect_trades(_inst())

        assert [t["profit_ratio"] for t in trades] == [-0.02, 0.03]
        assert client.get.call_args_list[0] == call("http://localhost:8090/api/v1/trades", params={"limit": 500})


class TestEndReason:
    def test_runs_on_below_both_limits(self):
        assert end_reason(_metrics(closed=TARGET_CLOSED_TRADES - 1), 0, MAX_RUN_DAYS * DAY - 1) is None

    def test_ends_at_target_closed_trades(self):
        assert end_reason(_metrics(closed=TARGET_CLOSED_TRADES), 0, DAY) == "30 closed trades"

    def test_open_trades_do_not_count(self):
        m = _metrics(closed=TARGET_CLOSED_TRADES - 1, trade_count=TARGET_CLOSED_TRADES + 2)
        assert end_reason(m, 0, DAY) is None

    def test_ends_at_max_days_even_without_metrics(self):
        assert end_reason(None, 0, MAX_RUN_DAYS * DAY) == "60 days"
        assert end_reason(None, 0, DAY) is None

    def test_limits_are_eddies_decision(self):
        assert (TARGET_CLOSED_TRADES, MAX_RUN_DAYS) == (30, 60)


class TestRunPaperArena:
    @patch("paper.monitor.finish_run")
    @patch("paper.monitor._poll")
    @patch("paper.monitor.time.sleep")
    @patch("paper.monitor.time.time")
    def test_each_run_ends_on_its_own_and_is_judged_on_its_last_poll(
        self, mock_time, mock_sleep, mock_poll, mock_finish
    ):
        a, b = _inst("StratA", 0), _inst("StratB", 1)
        # poll 1: neither done; poll 2: A hits 30 closed trades; poll 3: B reaches 60 days
        answers = iter([
            [_metrics("StratA", 10), _metrics("StratB", 2)],
            [_metrics("StratA", 30), _metrics("StratB", 3)],
            [_metrics("StratB", 4)],
        ])
        polled = []
        mock_poll.side_effect = lambda insts, db: polled.append(list(insts)) or next(answers)
        mock_time.side_effect = [DAY, 2 * DAY, MAX_RUN_DAYS * DAY]
        mock_finish.side_effect = lambda db, inst, m, s, e: {"passed": inst.strategy_name == "StratA"}

        results = run_paper_arena([a, b], started_at=0, db_url="postgresql://x")

        assert results == {"StratA": {"passed": True}, "StratB": {"passed": False}}
        finished = [(c.args[1].strategy_name, c.args[2]["closed_trade_count"], c.args[4]) for c in mock_finish.call_args_list]
        assert finished == [("StratA", 30, 2 * DAY), ("StratB", 4, MAX_RUN_DAYS * DAY)]
        assert mock_sleep.call_args_list == [call(POLL_INTERVAL_SECS)] * 2
        assert polled == [[a, b], [a, b], [b]]  # a finished run is no longer polled

    @patch("paper.monitor.finish_run", return_value={"passed": False})
    @patch("paper.monitor._write_metrics_snapshot")
    @patch("paper.monitor.collect_metrics", side_effect=RuntimeError("down"))
    @patch("paper.monitor.time.sleep")
    @patch("paper.monitor.time.time", side_effect=[DAY, MAX_RUN_DAYS * DAY])
    def test_unreachable_instance_fails_at_deadline_with_no_metrics(
        self, mock_time, mock_sleep, mock_collect, mock_write, mock_finish
    ):
        run_paper_arena([_inst()], started_at=0, db_url="postgresql://x")

        assert mock_finish.call_args.args[2] is None
        mock_write.assert_not_called()

    @patch("paper.monitor.finish_run", return_value={"passed": True})
    @patch("paper.monitor._write_metrics_snapshot")
    @patch("paper.monitor.collect_metrics")
    @patch("paper.monitor.time.sleep")
    @patch("paper.monitor.time.time", side_effect=[DAY, 2 * DAY])
    def test_writes_every_hourly_poll_to_db(self, mock_time, mock_sleep, mock_collect, mock_write, mock_finish):
        mock_collect.side_effect = [_metrics(closed=5), _metrics(closed=30)]

        run_paper_arena([_inst()], started_at=0, db_url="postgresql://x")

        assert mock_write.call_count == 2


class TestFinishRun:
    @patch("paper.monitor._record_result")
    @patch("paper.monitor.save_result")
    @patch("paper.monitor.evaluate_paper_run")
    @patch("paper.monitor.teardown_paper_instance")
    @patch("paper.monitor.collect_trades")
    def test_stops_instance_then_compares_and_records(
        self, mock_trades, mock_teardown, mock_eval, mock_save, mock_record, capsys
    ):
        order = []
        mock_trades.side_effect = lambda i: order.append("trades") or [{"profit_ratio": 0.01}]
        mock_teardown.side_effect = lambda i: order.append("teardown")
        mock_eval.side_effect = lambda *a: order.append("eval") or {
            "passed": True, "reason": "matches backtest", "strategy": "StratA"}
        inst, m = _inst(), _metrics(closed=30)

        result = finish_run("postgresql://x", inst, m, 100.0, 200.0)

        assert order == ["trades", "teardown", "eval"]
        mock_eval.assert_called_once_with("StratA", 100.0, 200.0, m, [{"profit_ratio": 0.01}])
        mock_save.assert_called_once_with(result)
        mock_record.assert_called_once_with("postgresql://x", "StratA", True)
        assert "PROMOTION CANDIDATE: StratA" in capsys.readouterr().out

    @patch("paper.monitor._record_result")
    @patch("paper.monitor.save_result")
    @patch("paper.monitor.evaluate_paper_run", return_value={"passed": False, "reason": "no final paper metrics or trades"})
    @patch("paper.monitor.teardown_paper_instance")
    @patch("paper.monitor.collect_trades", side_effect=RuntimeError("down"))
    def test_trade_fetch_failure_is_a_recorded_fail(
        self, mock_trades, mock_teardown, mock_eval, mock_save, mock_record, capsys
    ):
        finish_run("postgresql://x", _inst(), _metrics(closed=30), 0, 1)

        assert mock_eval.call_args.args[4] is None
        mock_record.assert_called_once_with("postgresql://x", "StratA", False)
        out = capsys.readouterr().out
        assert "PAPER FAIL: StratA" in out and "PROMOTION CANDIDATE" not in out


class TestRunCohort:
    @patch("paper.monitor.teardown_paper_instance")
    @patch("paper.monitor._mark_started")
    @patch("paper.monitor.run_paper_arena")
    @patch("paper.monitor.wait_until_ready", return_value=True)
    @patch("paper.monitor.spawn_paper_instance")
    @patch("paper.monitor.time.time", return_value=1234.0)
    def test_runs_arena_from_ready_time_and_tears_down(
        self, mock_time, mock_spawn, mock_ready, mock_arena, mock_started, mock_teardown
    ):
        from paper.monitor import run_cohort

        insts = [_inst("StratA", 0), _inst("StratB", 1)]
        mock_spawn.side_effect = insts

        run_cohort("postgresql://x", ["StratA", "StratB"])

        assert mock_spawn.call_args_list == [call("StratA", 0), call("StratB", 1)]
        mock_started.assert_called_once_with("postgresql://x", ["StratA", "StratB"])
        mock_arena.assert_called_once_with(insts, started_at=1234.0, db_url="postgresql://x")
        assert mock_teardown.call_count == 2

    @patch("paper.monitor.teardown_paper_instance")
    @patch("paper.monitor._mark_started")
    @patch("paper.monitor.run_paper_arena", side_effect=RuntimeError("boom"))
    @patch("paper.monitor.wait_until_ready", return_value=True)
    @patch("paper.monitor.spawn_paper_instance")
    def test_tears_down_when_arena_fails(self, mock_spawn, mock_ready, mock_arena, mock_started, mock_teardown):
        from paper.monitor import run_cohort

        mock_spawn.side_effect = [_inst("StratA", 0)]

        with pytest.raises(RuntimeError):
            run_cohort("postgresql://x", ["StratA"])

        mock_teardown.assert_called_once()
