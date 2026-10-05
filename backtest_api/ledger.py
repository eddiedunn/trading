"""Attempt ledger: what the backtest API has run, per campaign.

A campaign is one holdout date (periods.campaign_id()). The ledger counts how
many distinct pieces of code reached Phase 1 in the campaign (the Phase 1 Sharpe
bar rises with that count), remembers which code passed Phase 2, and allows each
piece of code, and each strategy name, exactly one final test on the held-back data.

One SQLite file, TRADING_RESULTS_DIR/ledger.sqlite, created on first use.
"""

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    id            INTEGER PRIMARY KEY,
    campaign      TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    code_sha256   TEXT NOT NULL,
    phase         INTEGER NOT NULL,
    passed        INTEGER,          -- NULL while the run is in progress or if it errored
    ts            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_campaign ON attempts (campaign, phase, code_sha256);

CREATE TABLE IF NOT EXISTS final_tests (
    id            INTEGER PRIMARY KEY,
    campaign      TEXT NOT NULL,
    code_sha256   TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    passed        INTEGER,          -- NULL while the test is running
    result_json   TEXT,
    ts            TEXT NOT NULL,
    UNIQUE (campaign, code_sha256),
    UNIQUE (campaign, strategy_name)
);
"""


class FinalTestAlreadyRun(Exception):
    """This code or this strategy name already has a final test in the campaign."""


def ledger_path() -> Path:
    results = Path(os.environ.get("TRADING_RESULTS_DIR", str(_REPO_ROOT / "backtest_results")))
    return results / "ledger.sqlite"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)  # autocommit; one statement per write
    conn.executescript(_SCHEMA)
    return conn


def code_sha256(code: str) -> str:
    """Hash of the code with trailing whitespace and trailing blank lines removed,
    so re-saving a file doesn't count as a new attempt but any real edit does."""
    normalised = "\n".join(line.rstrip() for line in code.splitlines()).rstrip("\n")
    return hashlib.sha256(normalised.encode()).hexdigest()


def record_attempt(campaign: str, strategy_name: str, sha: str, phase: int, passed: bool | None = None) -> int:
    """Log a Phase 1 or Phase 2 run. Returns the row id, for finish_attempt()."""
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO attempts (campaign, strategy_name, code_sha256, phase, passed, ts) VALUES (?, ?, ?, ?, ?, ?)",
            (campaign, strategy_name, sha, phase, None if passed is None else int(passed), _now()),
        )
        return cur.lastrowid
    finally:
        conn.close()


def finish_attempt(attempt_id: int, passed: bool | None) -> None:
    conn = _connect()
    try:
        conn.execute("UPDATE attempts SET passed = ? WHERE id = ?", (None if passed is None else int(passed), attempt_id))
    finally:
        conn.close()


def attempt_count(campaign: str) -> int:
    """Distinct pieces of code that reached Phase 1 in this campaign."""
    conn = _connect()
    try:
        return conn.execute(
            "SELECT COUNT(DISTINCT code_sha256) FROM attempts WHERE campaign = ? AND phase = 1", (campaign,)
        ).fetchone()[0]
    finally:
        conn.close()


def phase2_passed(campaign: str, sha: str) -> bool:
    conn = _connect()
    try:
        return conn.execute(
            "SELECT 1 FROM attempts WHERE campaign = ? AND code_sha256 = ? AND phase = 2 AND passed = 1 LIMIT 1",
            (campaign, sha),
        ).fetchone() is not None
    finally:
        conn.close()


def final_test_done(campaign: str, sha: str | None = None, name: str | None = None) -> bool:
    """True if this code, or this strategy name, has a final test (finished or running) this campaign."""
    conn = _connect()
    try:
        return conn.execute(
            "SELECT 1 FROM final_tests WHERE campaign = ? AND (code_sha256 = ? OR strategy_name = ?) LIMIT 1",
            (campaign, sha, name),
        ).fetchone() is not None
    finally:
        conn.close()


def claim_final_test(campaign: str, sha: str, name: str) -> None:
    """Reserve the one final test for this code and name before running it, so two
    concurrent requests can't both see the held-back data. Raises FinalTestAlreadyRun."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO final_tests (campaign, code_sha256, strategy_name, ts) VALUES (?, ?, ?, ?)",
            (campaign, sha, name, _now()),
        )
    except sqlite3.IntegrityError as e:
        raise FinalTestAlreadyRun(f"{name} or this code already had a final test in campaign {campaign}") from e
    finally:
        conn.close()


def release_final_test(campaign: str, sha: str) -> None:
    """Drop an unfinished claim (the final test crashed before producing a result)."""
    conn = _connect()
    try:
        conn.execute("DELETE FROM final_tests WHERE campaign = ? AND code_sha256 = ? AND passed IS NULL", (campaign, sha))
    finally:
        conn.close()


def record_final_test(campaign: str, sha: str, name: str, passed: bool, result: dict) -> None:
    """Store the final test result, filling in a claim or inserting a new row."""
    conn = _connect()
    try:
        cur = conn.execute(
            "UPDATE final_tests SET passed = ?, result_json = ?, ts = ? "
            "WHERE campaign = ? AND code_sha256 = ? AND strategy_name = ? AND passed IS NULL",
            (int(passed), json.dumps(result, default=str), _now(), campaign, sha, name),
        )
        if cur.rowcount == 0:
            try:
                conn.execute(
                    "INSERT INTO final_tests (campaign, code_sha256, strategy_name, passed, result_json, ts) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (campaign, sha, name, int(passed), json.dumps(result, default=str), _now()),
                )
            except sqlite3.IntegrityError as e:
                raise FinalTestAlreadyRun(f"{name} or this code already had a final test in campaign {campaign}") from e
    finally:
        conn.close()
