"""Paper arena monitor — collect metrics from running paper instances.

Polls Freqtrade REST APIs, writes snapshots to Postgres, evaluates
promotion criteria after the evaluation window.

Run as a service with ``python -m paper.monitor``: it picks up strategies
queued with ``trading_client paper-add``, runs them as one cohort for the
evaluation window, records the outcome, and waits for the next queue.
"""

import os
import time

import httpx
import psycopg2

from paper.orchestrator import (
    MAX_SLOTS,
    PaperInstance,
    spawn_paper_instance,
    teardown_all,
    teardown_paper_instance,
)

EVAL_WINDOW_DAYS = 14
POLL_INTERVAL_SECS = 3600  # check every hour

PROMOTION_CRITERIA = {
    "min_trades": 20,
    "min_profit_pct": 5.0,  # % total return over window
    "max_drawdown_pct": -15.0,
    "min_profit_factor": 1.25,
}


def collect_metrics(instance: PaperInstance) -> dict:
    """Collect current metrics from a paper instance's REST API."""
    base = f"http://localhost:{instance.port}"
    auth = (os.environ.get("FREQTRADE_API_USER", "freqtrade"), os.environ["FREQTRADE_API_PASSWORD"])

    with httpx.Client(auth=auth, timeout=30) as client:
        profit = client.get(f"{base}/api/v1/profit").json()
        status = client.get(f"{base}/api/v1/status").json()

    return {
        "strategy": instance.strategy_name,
        "profit_pct": profit.get("profit_all_percent") or 0,
        "trade_count": profit.get("trade_count") or 0,
        "win_rate": profit.get("winrate") or 0,
        # Freqtrade returns null until there is a losing trade
        "profit_factor": profit.get("profit_factor") or 0,
        # Freqtrade reports a positive fraction (0.12); criteria use negative percent (-12.0)
        "max_drawdown": -abs(profit.get("max_drawdown") or 0) * 100,
        "open_trades": len(status) if isinstance(status, list) else 0,
    }


def meets_promotion_criteria(metrics: dict) -> bool:
    """Check if a paper instance meets promotion thresholds.

    Win rate is recorded but not gated, the same as in Phase 1.
    """
    c = PROMOTION_CRITERIA
    return (
        metrics["trade_count"] >= c["min_trades"]
        and metrics["profit_pct"] >= c["min_profit_pct"]
        and metrics["max_drawdown"] >= c["max_drawdown_pct"]
        and metrics["profit_factor"] >= c["min_profit_factor"]
    )


def _poll(instances: list[PaperInstance], db_url: str | None) -> list[dict]:
    """Collect metrics from every instance once and store the snapshot."""
    all_metrics = []
    for inst in instances:
        try:
            all_metrics.append(collect_metrics(inst))
        except Exception as e:
            print(f"  Warning: failed to collect metrics from {inst.strategy_name}: {e}")
    if db_url and all_metrics:
        _write_metrics_snapshot(all_metrics, db_url)
    return all_metrics


def best_candidate(all_metrics: list[dict]) -> dict | None:
    """The highest profit factor among metrics that meet the criteria, or None."""
    candidates = [m for m in all_metrics if meets_promotion_criteria(m)]
    return max(candidates, key=lambda m: m["profit_factor"]) if candidates else None


def run_paper_arena(
    instances: list[PaperInstance],
    eval_days: int = EVAL_WINDOW_DAYS,
    db_url: str | None = None,
) -> list[dict]:
    """
    Poll all paper instances every hour for eval_days, then poll once more.

    Returns the metrics from that last poll. Only the last poll decides who
    passes: a strategy that met the bar mid-window and has since fallen below
    it is not a candidate. The promotion line is printed from the same metrics.
    """
    deadline = time.time() + eval_days * 86400

    while time.time() < deadline:
        leader = best_candidate(_poll(instances, db_url))
        if leader:
            print(
                f"  -> Meets criteria at this poll: {leader['strategy']} "
                f"PF={leader['profit_factor']:.2f} "
                f"WR={leader['win_rate']:.1%} "
                f"Return={leader['profit_pct']:.1f}%"
            )
        time.sleep(POLL_INTERVAL_SECS)

    final = _poll(instances, db_url)
    best = best_candidate(final)
    if best:
        print(
            f"PROMOTION CANDIDATE: {best['strategy']} — "
            f"run `python -m live.trading_client promote --strategy {best['strategy']}` to activate"
        )
    return final


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


STARTUP_TIMEOUT_SECS = 180
IDLE_POLL_SECS = 300


def db_url_from_env() -> str:
    return "postgresql://{user}:{password}@{host}:{port}/{db}".format(
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ.get("POSTGRES_HOST", "127.0.0.1"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        db=os.environ["POSTGRES_DB"],
    )


def _execute(db_url: str, sql: str, params: tuple = ()) -> list[tuple]:
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() if cur.description else []
        conn.commit()
        return rows
    finally:
        conn.close()


def queued_strategies(db_url: str, limit: int = MAX_SLOTS) -> list[str]:
    """Strategies queued for paper trading that have not started yet, oldest first."""
    rows = _execute(
        db_url,
        """
        SELECT name FROM strategy_registry
         WHERE paper_queued_at IS NOT NULL AND paper_started_at IS NULL
         ORDER BY paper_queued_at
         LIMIT %s
        """,
        (limit,),
    )
    return [r[0] for r in rows]


def requeue_interrupted(db_url: str) -> None:
    """A restart loses the running cohort; put its strategies back in the queue."""
    _execute(
        db_url,
        """
        UPDATE strategy_registry SET paper_started_at = NULL
         WHERE paper_started_at IS NOT NULL AND paper_finished_at IS NULL
        """,
    )


def _mark_started(db_url: str, names: list[str]) -> None:
    _execute(
        db_url,
        "UPDATE strategy_registry SET paper_started_at = NOW() WHERE name = ANY(%s) AND paper_started_at IS NULL",
        (names,),
    )


def _record_result(db_url: str, name: str, passed: bool) -> None:
    _execute(
        db_url,
        """
        UPDATE strategy_registry
           SET paper_passed = %s, paper_finished_at = NOW()
         WHERE name = %s AND paper_finished_at IS NULL
        """,
        (passed, name),
    )


def wait_until_ready(instance: PaperInstance, timeout: float = STARTUP_TIMEOUT_SECS) -> bool:
    """Wait for a fresh Freqtrade container to answer /api/v1/ping."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://localhost:{instance.port}/api/v1/ping", timeout=5)
            if r.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(5)
    return False


def run_cohort(db_url: str, names: list[str], eval_days: int = EVAL_WINDOW_DAYS) -> None:
    """Paper-trade one cohort for the evaluation window and record each outcome."""
    instances = [spawn_paper_instance(name, slot) for slot, name in enumerate(names)]
    _mark_started(db_url, names)
    try:
        for inst in instances:
            if not wait_until_ready(inst):
                print(f"  Warning: {inst.container_name} did not answer /api/v1/ping in time")
        final = {m["strategy"]: m for m in run_paper_arena(instances, eval_days=eval_days, db_url=db_url)}
        for inst in instances:
            metrics = final.get(inst.strategy_name)
            if metrics is None:
                print(f"  Warning: no final metrics for {inst.strategy_name}; recording a fail")
            passed = metrics is not None and meets_promotion_criteria(metrics)
            _record_result(db_url, inst.strategy_name, passed)
    finally:
        for inst in instances:
            teardown_paper_instance(inst)


def main() -> None:
    db_url = db_url_from_env()
    teardown_all()
    requeue_interrupted(db_url)
    print("Paper arena monitor started; waiting for queued strategies.")
    while True:
        names = queued_strategies(db_url)
        if names:
            print(f"Starting paper cohort: {', '.join(names)}")
            run_cohort(db_url, names)
        else:
            time.sleep(IDLE_POLL_SECS)


if __name__ == "__main__":
    main()
