"""Attempt ledger: what the backtest API has run, per campaign.

A campaign is one holdout date (periods.campaign_id()). The ledger counts the
effective number of trials that reached Phase 1 in the campaign (reported, see
attempt_counts()), counts final tests (the final test's Sharpe bar rises with
that count), remembers which code passed Phase 2, and allows each
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


# How much a revision of an existing idea (same strategy name, new code) adds to the
# effective trial count, relative to a brand-new idea.
#
# The multiple-testing bar (fast_filter.expected_max_sharpe, Bailey & Lopez de Prado)
# assumes N *independent* trials. Revisions of one idea share its signal, its
# indicators and most of its parameters, so their returns are highly correlated:
# counting each as a full trial overstates N and pushes the bar up too fast (a real
# run went 0.80 -> 1.66 after 2 ideas x 3 revisions). Each revision still is a
# search step, though, so it can't count as zero either. 0.25 is a judgement call,
# not an estimate of the actual correlation.
#
# Limits: the count trusts the strategy name. A human could rename every revision to
# make it count fully (raising their own bar, harmless) or reuse one name for
# unrelated ideas to make them count as revisions (gaming it down). The agent names
# each idea once and keeps the name across revisions (agent/loop.py), so for agent
# runs the name is a fair idea label.
REVISION_WEIGHT = 0.25


def attempt_counts(campaign: str) -> dict:
    """Trials that reached Phase 1 in this campaign.

    - ``versions``: distinct code hashes, from any strategy.
    - ``ideas``: distinct strategy names that brought new code, i.e. were the
      first to submit at least one of those hashes. Normally that's every name; a
      name that only re-submitted code another name already ran is not a new idea.
    - ``effective``: ideas + REVISION_WEIGHT * (versions - ideas), the N used for
      the Sharpe bar. 0.0 before anything is tried, else >= 1, and identical code
      under any name never adds to it.

    Computed from the existing ``attempts`` columns; no schema change.
    """
    conn = _connect()
    try:
        ideas, versions = conn.execute(
            """
            WITH first_run AS (
                SELECT code_sha256, MIN(id) AS id FROM attempts
                WHERE campaign = ? AND phase = 1 GROUP BY code_sha256
            )
            SELECT COUNT(DISTINCT a.strategy_name), COUNT(*)
            FROM first_run f JOIN attempts a ON a.id = f.id
            """,
            (campaign,),
        ).fetchone()
    finally:
        conn.close()
    return {"ideas": ideas, "versions": versions, "effective": float(ideas + REVISION_WEIGHT * (versions - ideas))}


def attempt_count(campaign: str) -> float:
    """Effective number of trials that reached Phase 1 in this campaign (see attempt_counts)."""
    return attempt_counts(campaign)["effective"]


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


def final_test_count(campaign: str) -> int:
    """Final tests run or running this campaign, including one just claimed."""
    conn = _connect()
    try:
        return conn.execute("SELECT COUNT(*) FROM final_tests WHERE campaign = ?", (campaign,)).fetchone()[0]
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
