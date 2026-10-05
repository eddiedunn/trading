"""Unit tests for live promotion CLI.

Tests the three main subcommands: promote, status, retire.
Each covers file operations, database interactions, and CLI argument parsing.
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

from live.trading_client import (
    promote,
    status,
    retire,
    main,
    _connect,
    _write_active,
    _copy_strategy,
    paper_add,
    LIVE_DIR,
    ACTIVE_FILE,
    STRATEGIES_DIR,
    NULL_STRATEGY,
)


class TestConnect:
    """Test database connection setup."""

    @patch.dict(
        os.environ,
        {
            "POSTGRES_HOST": "localhost",
            "POSTGRES_DB": "trading",
            "POSTGRES_USER": "user",
            "POSTGRES_PASSWORD": "pass",
        },
    )
    @patch("live.trading_client.psycopg2.connect")
    def test_connect_uses_env_vars(self, mock_connect):
        """Verify psycopg2.connect is called with env var credentials."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        result = _connect()

        assert result == mock_conn
        mock_connect.assert_called_once_with(
            host="localhost",
            dbname="trading",
            user="user",
            password="pass",
        )


class TestWriteActive:
    """Test active_strategy.txt file writing."""

    @patch("live.trading_client.LIVE_DIR")
    @patch("live.trading_client.ACTIVE_FILE")
    def test_write_active_creates_directory(self, mock_active_file, mock_live_dir):
        """Verify LIVE_DIR is created with parents=True, exist_ok=True."""
        _write_active("TestStrat")

        mock_live_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("live.trading_client.LIVE_DIR")
    @patch("live.trading_client.ACTIVE_FILE")
    def test_write_active_writes_name_with_newline(self, mock_active_file, mock_live_dir):
        """Verify file content is 'name\\n'."""
        _write_active("MyStrategy")

        mock_live_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_active_file.write_text.assert_called_once_with("MyStrategy\n")

    @patch("live.trading_client.LIVE_DIR")
    @patch("live.trading_client.ACTIVE_FILE")
    def test_write_active_null_strategy(self, mock_active_file, mock_live_dir):
        """Verify NullStrategy can be written."""
        _write_active(NULL_STRATEGY)

        mock_active_file.write_text.assert_called_once_with("NullStrategy\n")


class TestCopyStrategy:
    """Test strategy file copying."""

    @pytest.fixture
    def dirs(self, tmp_path, monkeypatch):
        strat = tmp_path / "strategies"
        live = tmp_path / "live"
        (strat / "candidates").mkdir(parents=True)
        (strat / "NullStrategy.py").write_text("null\n")
        monkeypatch.setattr("live.trading_client.STRATEGIES_DIR", strat)
        monkeypatch.setattr("live.trading_client.LIVE_DIR", live)
        return strat, live

    def test_copy_from_strategies_dir(self, dirs):
        """Copy strategy from strategies/<name>.py when it exists."""
        strat, live = dirs
        (strat / "TestStrat.py").write_text("top\n")

        _copy_strategy("TestStrat")

        assert (live / "TestStrat.py").read_text() == "top\n"

    def test_copy_from_candidates_fallback(self, dirs):
        """Fall back to candidates/ when strategies/<name>.py not found."""
        strat, live = dirs
        (strat / "candidates" / "TestStrat.py").write_text("cand\n")

        _copy_strategy("TestStrat")

        assert (live / "TestStrat.py").read_text() == "cand\n"

    def test_copy_cleans_stale_py_files(self, dirs):
        """Stale .py files in LIVE_DIR are removed before copy."""
        strat, live = dirs
        (strat / "TestStrat.py").write_text("new\n")
        live.mkdir()
        (live / "OldStrat.py").write_text("old\n")
        (live / "config.json").write_text("{}")

        _copy_strategy("TestStrat")

        assert not (live / "OldStrat.py").exists()
        assert (live / "config.json").exists()

    def test_copy_keeps_null_strategy(self, dirs):
        """NullStrategy.py stays in the live slot so retire can fall back to it."""
        strat, live = dirs
        (strat / "TestStrat.py").write_text("new\n")
        live.mkdir()
        (live / "NullStrategy.py").write_text("live null\n")

        _copy_strategy("TestStrat")

        assert (live / "NullStrategy.py").read_text() == "live null\n"

    def test_copy_restores_missing_null_strategy(self, dirs):
        """A live slot that lost NullStrategy.py gets it back from strategies/."""
        strat, live = dirs
        (strat / "TestStrat.py").write_text("new\n")

        _copy_strategy("TestStrat")

        assert (live / "NullStrategy.py").read_text() == "null\n"

    def test_copy_fails_when_file_not_found(self, dirs):
        """Missing source raises before the live slot is touched."""
        _, live = dirs
        live.mkdir()
        (live / "OldStrat.py").write_text("old\n")

        with pytest.raises(FileNotFoundError):
            _copy_strategy("NonExistent")

        assert (live / "OldStrat.py").exists()

class TestPromote:
    """Test the promote subcommand."""

    @patch("live.trading_client._connect")
    @patch("live.trading_client._write_active")
    @patch("live.trading_client._copy_strategy")
    def test_promote_copies_writes_updates_db(
        self, mock_copy, mock_write, mock_connect
    ):
        """Full promote flow: copy file, write active, update DB."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        promote("TestStrat")

        # Verify sequence
        mock_copy.assert_called_once_with("TestStrat")
        mock_write.assert_called_once_with("TestStrat")
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("live.trading_client._connect")
    @patch("live.trading_client._write_active")
    @patch("live.trading_client._copy_strategy")
    def test_promote_sql_shape(self, mock_copy, mock_write, mock_connect):
        """Verify UPDATE statement has correct WHERE and SET clauses."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        promote("TestStrat")

        # Check SQL statement
        sql, params = mock_cursor.execute.call_args[0]
        assert "UPDATE strategy_registry" in sql
        assert "promoted_live = TRUE" in sql
        assert "promoted_at = NOW()" in sql
        assert "WHERE name = %s" in sql
        assert params == ("TestStrat",)

    @patch("live.trading_client._connect")
    @patch("live.trading_client._write_active")
    @patch("live.trading_client._copy_strategy")
    def test_promote_closes_connection_on_success(
        self, mock_copy, mock_write, mock_connect
    ):
        """Verify connection is closed after successful promote."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        promote("TestStrat")

        mock_conn.close.assert_called_once()

    @patch("live.trading_client._connect")
    @patch("live.trading_client._write_active")
    @patch("live.trading_client._copy_strategy")
    def test_promote_closes_connection_on_error(
        self, mock_copy, mock_write, mock_connect
    ):
        """Verify connection is closed even if DB error occurs."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.execute.side_effect = Exception("DB error")

        with pytest.raises(Exception):
            promote("TestStrat")

        mock_conn.close.assert_called_once()

    @patch("live.trading_client._connect")
    @patch("live.trading_client._write_active")
    @patch("live.trading_client._copy_strategy")
    def test_promote_prints_success_message(
        self, mock_copy, mock_write, mock_connect, capsys
    ):
        """Verify success message is printed."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        promote("TestStrat")

        captured = capsys.readouterr()
        assert "Promoted TestStrat to live slot" in captured.out
        assert "active_strategy.txt updated" in captured.out

    @patch("live.trading_client._connect")
    @patch("live.trading_client._write_active")
    @patch("live.trading_client._copy_strategy")
    def test_promote_with_candidates_fallback(
        self, mock_copy, mock_write, mock_connect, capsys
    ):
        """Promote works even if strategy is in candidates/ fallback."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        promote("CandidateStrat")

        mock_copy.assert_called_once_with("CandidateStrat")
        mock_write.assert_called_once_with("CandidateStrat")

        captured = capsys.readouterr()
        assert "Promoted CandidateStrat" in captured.out


class TestStatus:
    """Test the status subcommand."""

    @patch("live.trading_client._connect")
    @patch("live.trading_client.ACTIVE_FILE")
    def test_status_reads_db_and_file(
        self, mock_active_file, mock_connect, capsys
    ):
        """Read promoted strategy from DB and active file content."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = ("TestStrat", "2024-01-01 10:00:00")
        mock_active_file.exists.return_value = True
        mock_active_file.read_text.return_value = "TestStrat"

        status()

        # Verify DB query
        sql = mock_cursor.execute.call_args[0][0]
        assert "SELECT name, promoted_at" in sql
        assert "FROM strategy_registry" in sql
        assert "WHERE promoted_live = TRUE" in sql
        assert "ORDER BY promoted_at DESC" in sql
        assert "LIMIT 1" in sql

        # Verify output
        captured = capsys.readouterr()
        assert "active_strategy.txt: TestStrat" in captured.out
        assert "registry promoted:   TestStrat at 2024-01-01 10:00:00" in captured.out

    @patch("live.trading_client._connect")
    @patch("live.trading_client.ACTIVE_FILE")
    def test_status_when_active_file_missing(
        self, mock_active_file, mock_connect, capsys
    ):
        """Show NullStrategy when active_strategy.txt doesn't exist."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = ("TestStrat", "2024-01-01 10:00:00")
        mock_active_file.exists.return_value = False

        status()

        captured = capsys.readouterr()
        assert "active_strategy.txt: NullStrategy" in captured.out

    @patch("live.trading_client._connect")
    @patch("live.trading_client.ACTIVE_FILE")
    def test_status_when_no_promoted(
        self, mock_active_file, mock_connect, capsys
    ):
        """Show 'none' when no promoted strategy in registry."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        mock_active_file.exists.return_value = False

        status()

        captured = capsys.readouterr()
        assert "active_strategy.txt: NullStrategy" in captured.out
        assert "registry promoted:   none" in captured.out

    @patch("live.trading_client._connect")
    @patch("live.trading_client.ACTIVE_FILE")
    def test_status_closes_connection(
        self, mock_active_file, mock_connect
    ):
        """Verify connection is closed after status."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        mock_active_file.exists.return_value = False

        status()

        mock_conn.close.assert_called_once()


class TestRetire:
    """Test the retire subcommand."""

    @patch("live.trading_client._connect")
    @patch("live.trading_client._write_active")
    def test_retire_writes_null_and_updates_db(self, mock_write, mock_connect):
        """Full retire flow: write NullStrategy, update DB."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        retire("TestStrat")

        # Verify sequence
        mock_write.assert_called_once_with(NULL_STRATEGY)
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("live.trading_client._connect")
    @patch("live.trading_client._write_active")
    def test_retire_sql_shape(self, mock_write, mock_connect):
        """Verify UPDATE sets retired_at=NOW() and promoted_live=FALSE."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        retire("TestStrat")

        sql, params = mock_cursor.execute.call_args[0]
        assert "UPDATE strategy_registry" in sql
        assert "retired_at = NOW()" in sql
        assert "promoted_live = FALSE" in sql
        assert "WHERE name = %s" in sql
        assert params == ("TestStrat",)

    @patch("live.trading_client._connect")
    @patch("live.trading_client._write_active")
    def test_retire_closes_connection(self, mock_write, mock_connect):
        """Verify connection is closed after retire."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        retire("TestStrat")

        mock_conn.close.assert_called_once()

    @patch("live.trading_client._connect")
    @patch("live.trading_client._write_active")
    def test_retire_prints_success_message(self, mock_write, mock_connect, capsys):
        """Verify success message is printed."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        retire("TestStrat")

        captured = capsys.readouterr()
        assert "Retired TestStrat" in captured.out
        assert f"live reverted to {NULL_STRATEGY}" in captured.out

    @patch("live.trading_client._connect")
    @patch("live.trading_client._write_active")
    def test_retire_closes_on_error(self, mock_write, mock_connect):
        """Verify connection is closed even if error occurs."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.execute.side_effect = Exception("DB error")

        with pytest.raises(Exception):
            retire("TestStrat")

        mock_conn.close.assert_called_once()


class TestMainCLI:
    """Test argparse wiring for main()."""

    @patch("live.trading_client.promote")
    def test_promote_subcommand(self, mock_promote):
        """Test promote subcommand invocation."""
        with patch("sys.argv", ["trading_client", "promote", "--strategy", "MyStrat"]):
            main()

        mock_promote.assert_called_once_with("MyStrat")

    @patch("live.trading_client.status")
    def test_status_subcommand(self, mock_status):
        """Test status subcommand invocation."""
        with patch("sys.argv", ["trading_client", "status"]):
            main()

        mock_status.assert_called_once()

    @patch("live.trading_client.retire")
    def test_retire_subcommand(self, mock_retire):
        """Test retire subcommand invocation."""
        with patch("sys.argv", ["trading_client", "retire", "--strategy", "MyStrat"]):
            main()

        mock_retire.assert_called_once_with("MyStrat")

    def test_promote_requires_strategy(self):
        """Test promote requires --strategy argument."""
        with patch("sys.argv", ["trading_client", "promote"]):
            with pytest.raises(SystemExit):
                main()

    def test_retire_requires_strategy(self):
        """Test retire requires --strategy argument."""
        with patch("sys.argv", ["trading_client", "retire"]):
            with pytest.raises(SystemExit):
                main()


class TestPaperAdd:
    """Test queueing a strategy for the paper arena."""

    @pytest.fixture(autouse=True)
    def _env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADING_PAPER_DIR", str(tmp_path / "paper"))
        self.code = tmp_path / "EmaCross.py"
        self.code.write_text("# strategy")
        self.paper = tmp_path / "paper"

    @patch("live.trading_client._connect")
    def test_queues_passing_strategy(self, mock_connect):
        cur = mock_connect.return_value.cursor.return_value.__enter__.return_value

        paper_add("EmaCross", self.code, {"passed": True}, {"passed": True})

        assert (self.paper / "strategies" / "EmaCross.py").read_text() == "# strategy"
        sql, params = cur.execute.call_args[0]
        assert "INSERT INTO strategy_registry" in sql
        assert "paper_queued_at" in sql
        assert params[0] == "EmaCross"
        assert params[1] is True and params[3] is True
        assert params[5] is None
        mock_connect.return_value.commit.assert_called_once()

    @patch("live.trading_client._connect")
    def test_refuses_failing_strategy_without_force(self, mock_connect):
        with pytest.raises(SystemExit):
            paper_add("EmaCross", self.code, {"passed": True}, {"passed": False})

        mock_connect.assert_not_called()
        assert not (self.paper / "strategies" / "EmaCross.py").exists()

    @patch("live.trading_client._connect")
    def test_force_queues_and_notes_it(self, mock_connect):
        cur = mock_connect.return_value.cursor.return_value.__enter__.return_value

        paper_add("EmaCross", self.code, {"passed": False}, {"passed": False}, force=True)

        params = cur.execute.call_args[0][1]
        assert params[1] is False and params[3] is False
        assert params[5] == "queued with --force"
