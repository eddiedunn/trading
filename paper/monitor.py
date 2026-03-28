"""Paper arena monitor — collect metrics from running paper instances.

Polls Freqtrade REST APIs, writes snapshots to Postgres, evaluates
promotion criteria after the evaluation window.
"""

import os
import time
from typing import Any

import httpx
import psycopg2

from paper.orchestrator import PaperInstance

EVAL_WINDOW_DAYS = 14
POLL_INTERVAL_SECS = 3600  # check every hour

PROMOTION_CRITERIA = {
    "min_trades": 20,
    "min_profit_pct": 5.0,  # % total return over window
    "max_drawdown_pct": -15.0,
    "min_win_rate": 0.45,
    "min_profit_factor": 1.25,
}


def collect_metrics(instance: PaperInstance) -> dict:
    """Collect current metrics from a paper instance's REST API."""
    base = f"http://localhost:{instance.port}"
    auth = ("freqtrade", "changeme")

    with httpx.Client(auth=auth, timeout=30) as client:
        profit = client.get(f"{base}/api/v1/profit").json()
        status = client.get(f"{base}/api/v1/status").json()

    return {
        "strategy": instance.strategy_name,
        "profit_pct": profit.get("profit_all_percent", 0),
        "trade_count": profit.get("trade_count", 0),
        "win_rate": profit.get("winrate", 0),
        "profit_factor": profit.get("profit_factor", 0),
        "max_drawdown": profit.get("max_drawdown", 0),
        "open_trades": len(status) if isinstance(status, list) else 0,
    }


def meets_promotion_criteria(metrics: dict) -> bool:
    """Check if a paper instance meets promotion thresholds."""
    c = PROMOTION_CRITERIA
    return (
        metrics["trade_count"] >= c["min_trades"]
        and metrics["profit_pct"] >= c["min_profit_pct"]
        and metrics["max_drawdown"] >= c["max_drawdown_pct"]
        and metrics["win_rate"] >= c["min_win_rate"]
        and metrics["profit_factor"] >= c["min_profit_factor"]
    )


def run_paper_arena(
    instances: list[PaperInstance],
    eval_days: int = EVAL_WINDOW_DAYS,
    db_url: str | None = None,
) -> dict | None:
    """
    Poll all paper instances every hour for eval_days.
    Return the best performer that meets criteria, or None.
    """
    deadline = time.time() + eval_days * 86400
    best = None

    while time.time() < deadline:
        all_metrics = []
        for inst in instances:
            try:
                all_metrics.append(collect_metrics(inst))
            except Exception as e:
                print(f"  Warning: failed to collect metrics from {inst.strategy_name}: {e}")

        if db_url and all_metrics:
            _write_metrics_snapshot(all_metrics, db_url)

        candidates = [m for m in all_metrics if meets_promotion_criteria(m)]
        if candidates:
            best = max(candidates, key=lambda m: m["profit_factor"])
            print(
                f"  -> Promotion candidate: {best['strategy']} "
                f"PF={best['profit_factor']:.2f} "
                f"WR={best['win_rate']:.1%} "
                f"Return={best['profit_pct']:.1f}%"
            )

        time.sleep(POLL_INTERVAL_SECS)

    return best


def _write_metrics_snapshot(all_metrics: list[dict], db_url: str):
    """Persist snapshot to Postgres for later analysis."""
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            for m in all_metrics:
                cur.execute(
                    """
                    INSERT INTO paper_snapshots
                      (ts, strategy, profit_pct, trade_count, win_rate, profit_factor, max_drawdown)
                    VALUES (NOW(), %(strategy)s, %(profit_pct)s, %(trade_count)s,
                            %(win_rate)s, %(profit_factor)s, %(max_drawdown)s)
                    """,
                    m,
                )
        conn.commit()
    finally:
        conn.close()
