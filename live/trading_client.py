"""Live promotion CLI — manually swap the live bot's active strategy."""

import argparse
import os
import shutil
from pathlib import Path

import psycopg2

LIVE_DIR = Path("/opt/trading/live")
ACTIVE_FILE = LIVE_DIR / "active_strategy.txt"
LIVE_SLOT = LIVE_DIR / "strategy.py"
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
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, LIVE_SLOT)


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
    print(f"Promoted {name} to live slot ({LIVE_SLOT}); active_strategy.txt updated.")


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

    pr = sub.add_parser("retire")
    pr.add_argument("--strategy", required=True)

    args = p.parse_args()
    if args.cmd == "promote":
        promote(args.strategy)
    elif args.cmd == "retire":
        retire(args.strategy)
    elif args.cmd == "status":
        status()


if __name__ == "__main__":
    main()
