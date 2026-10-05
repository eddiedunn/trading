"""Live promotion CLI — manually swap the live bot's active strategy."""

import argparse
import json
import os
import shutil
from pathlib import Path

import psycopg2
from psycopg2.extras import Json

from paper.orchestrator import strategies_dir as paper_strategies_dir

LIVE_DIR = Path("/opt/trading/live")
ACTIVE_FILE = LIVE_DIR / "active_strategy.txt"
STRATEGIES_DIR = Path("/opt/trading/strategies")
NULL_STRATEGY = "NullStrategy"


def _connect():
    return psycopg2.connect(
        host=os.environ["POSTGRES_HOST"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


def _write_active(name: str):
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_FILE.write_text(name + "\n")


def _copy_strategy(name: str):
    src = STRATEGIES_DIR / f"{name}.py"
    if not src.exists():
        src = STRATEGIES_DIR / "candidates" / f"{name}.py"
    # Check before touching the live slot, so a typo can't leave it empty.
    if not src.exists():
        raise FileNotFoundError(f"No strategy file for {name} in {STRATEGIES_DIR} or its candidates/")
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    # Keep NullStrategy.py in the live slot so retire can fall back to it.
    null_name = f"{NULL_STRATEGY}.py"
    for stale in LIVE_DIR.glob("*.py"):
        if stale.name != null_name:
            stale.unlink()
    shutil.copy2(src, LIVE_DIR / f"{name}.py")
    if not (LIVE_DIR / null_name).exists() and (STRATEGIES_DIR / null_name).exists():
        shutil.copy2(STRATEGIES_DIR / null_name, LIVE_DIR / null_name)


def promote(name: str):
    _copy_strategy(name)
    _write_active(name)
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE strategy_registry
                   SET promoted_live = TRUE, promoted_at = NOW()
                 WHERE name = %s
                """,
                (name,),
            )
        conn.commit()
    finally:
        conn.close()
    print(f"Promoted {name} to live slot ({LIVE_DIR / f'{name}.py'}); active_strategy.txt updated.")


def retire(name: str):
    _write_active(NULL_STRATEGY)
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE strategy_registry
                   SET retired_at = NOW(), promoted_live = FALSE
                 WHERE name = %s
                """,
                (name,),
            )
        conn.commit()
    finally:
        conn.close()
    print(f"Retired {name}; live reverted to {NULL_STRATEGY}.")


def paper_add(name: str, code_path: Path, phase1: dict | None, phase2: dict | None, force: bool = False):
    """Queue a backtested strategy for the paper arena.

    phase1 / phase2 are the /backtest API responses. Both must have passed
    unless force is set (used to exercise the pipeline with a losing example).
    """
    p1 = bool(phase1 and phase1.get("passed"))
    p2 = bool(phase2 and phase2.get("passed"))
    if not (p1 and p2) and not force:
        raise SystemExit(f"{name} has not passed Phase 1 and Phase 2 (phase1={p1}, phase2={p2}); pass --force to queue anyway")

    dest_dir = paper_strategies_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(code_path, dest_dir / f"{name}.py")

    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO strategy_registry
                  (name, phase1_passed, phase1_stats, phase2_passed, phase2_stats, paper_queued_at, notes)
                VALUES (%s, %s, %s, %s, %s, NOW(), %s)
                """,
                (
                    name,
                    p1,
                    Json(phase1) if phase1 else None,
                    p2,
                    Json(phase2) if phase2 else None,
                    "queued with --force" if force and not (p1 and p2) else None,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    print(f"Queued {name} for the paper arena ({dest_dir / f'{name}.py'}).")


def _load_json(path: str | None) -> dict | None:
    return json.loads(Path(path).read_text()) if path else None


def status():
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT name, promoted_at
                  FROM strategy_registry
                 WHERE promoted_live = TRUE
                 ORDER BY promoted_at DESC
                 LIMIT 1
                """
            )
            row = cur.fetchone()
    finally:
        conn.close()

    active = ACTIVE_FILE.read_text().strip() if ACTIVE_FILE.exists() else NULL_STRATEGY
    if row:
        print(f"active_strategy.txt: {active}")
        print(f"registry promoted:   {row[0]} at {row[1]}")
    else:
        print(f"active_strategy.txt: {active}")
        print("registry promoted:   none")


def main():
    p = argparse.ArgumentParser(prog="trading_client")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("promote")
    pp.add_argument("--strategy", required=True)

    sub.add_parser("status")

    pa = sub.add_parser("paper-add", help="queue a backtested strategy for the paper arena")
    pa.add_argument("--strategy", required=True)
    pa.add_argument("--file", required=True, help="strategy .py file")
    pa.add_argument("--phase1", help="Phase 1 /backtest response JSON file")
    pa.add_argument("--phase2", help="Phase 2 /backtest response JSON file")
    pa.add_argument("--force", action="store_true", help="queue even if a phase did not pass")

    pr = sub.add_parser("retire")
    pr.add_argument("--strategy", required=True)

    args = p.parse_args()
    if args.cmd == "promote":
        promote(args.strategy)
    elif args.cmd == "retire":
        retire(args.strategy)
    elif args.cmd == "status":
        status()
    elif args.cmd == "paper-add":
        paper_add(args.strategy, Path(args.file), _load_json(args.phase1), _load_json(args.phase2), args.force)


if __name__ == "__main__":
    main()
