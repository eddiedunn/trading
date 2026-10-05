"""Paper arena monitor — collect metrics from running paper instances.

Polls Freqtrade REST APIs every hour and writes snapshots to Postgres. A
strategy's paper run ends at ``TARGET_CLOSED_TRADES`` closed trades or after
``MAX_RUN_DAYS``, whichever comes first. It then passes if paper matched a
backtest of the same strategy over the same period (see ``paper.compare``).

Run as a service with ``python -m paper.monitor``: it picks up strategies
queued with ``trading_client paper-add``, runs them as one cohort until every
run has ended, records each outcome, and waits for the next queue.
"""

import json
import os
import time

import httpx
import psycopg2

from paper.compare import evaluate_paper_run, save_result
from paper.orchestrator import (
    MAX_SLOTS,
    PaperInstance,
    spawn_paper_instance,
    teardown_all,
    teardown_paper_instance,
)

TARGET_CLOSED_TRADES = 30
MAX_RUN_DAYS = 60
POLL_INTERVAL_SECS = 3600  # snapshot every hour


def _auth() -> tuple[str, str]:
    return (os.environ.get("FREQTRADE_API_USER", "freqtrade"), os.environ["FREQTRADE_API_PASSWORD"])


def collect_metrics(instance: PaperInstance) -> dict:
    """Collect current metrics from a paper instance's REST API."""
    base = f"http://localhost:{instance.port}"

    with httpx.Client(auth=_auth(), timeout=30) as client:
        profit = client.get(f"{base}/api/v1/profit").json()
        status = client.get(f"{base}/api/v1/status").json()

    return {
        "strategy": instance.strategy_name,
        # Closed plus open trades, valued from the starting wallet
        "profit_pct": profit.get("profit_all_percent") or 0,
        "trade_count": profit.get("trade_count") or 0,
        "closed_trade_count": profit.get("closed_trade_count") or 0,
        "win_rate": profit.get("winrate") or 0,
        # Freqtrade returns null until there is a losing trade
        "profit_factor": profit.get("profit_factor") or 0,
        # Freqtrade reports a positive fraction (0.12); stored as negative percent (-12.0)
        "max_drawdown": -abs(profit.get("max_drawdown") or 0) * 100,
        "open_trades": len(status) if isinstance(status, list) else 0,
    }


def collect_trades(instance: PaperInstance) -> list[dict]:
    """Closed trades plus open ones at their current value (``profit_ratio``, ``profit_abs``)."""
    base = f"http://localhost:{instance.port}"
    with httpx.Client(auth=_auth(), timeout=30) as client:
        closed = client.get(f"{base}/api/v1/trades", params={"limit": 500}).json()
        status = client.get(f"{base}/api/v1/status").json()
    trades = list(closed.get("trades") or [])
    if isinstance(status, list):
        trades.extend(status)
    return trades


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


def end_reason(metrics: dict | None, started_at: float, now: float) -> str | None:
    """Why this run is over, or None while it continues."""
    if metrics and metrics.get("closed_trade_count", 0) >= TARGET_CLOSED_TRADES:
        return f"{TARGET_CLOSED_TRADES} closed trades"
    if now - started_at >= MAX_RUN_DAYS * 86400:
        return f"{MAX_RUN_DAYS} days"
    return None


def finish_run(
    db_url: str | None, inst: PaperInstance, metrics: dict | None, started_at: float, ended_at: float
) -> dict:
    """Stop the instance, compare its final results with a backtest, and record pass/fail."""
    trades = None
    if metrics is not None:
        try:
            trades = collect_trades(inst)
        except Exception as e:
            print(f"  Warning: failed to collect trades from {inst.strategy_name}: {e}")
    teardown_paper_instance(inst)

    result = evaluate_paper_run(inst.strategy_name, started_at, ended_at, metrics, trades)
    try:
        path = save_result(result)
        print(f"  Comparison saved to {path}")
    except OSError as e:
        print(f"  Warning: could not save comparison for {inst.strategy_name}: {e}")
    print(f"PAPER RESULT {json.dumps(result, default=str)}")
    if result["passed"]:
        print(
            f"PROMOTION CANDIDATE: {inst.strategy_name} — "
            f"run `python -m live.trading_client promote --strategy {inst.strategy_name}` to activate"
        )
    else:
        print(f"PAPER FAIL: {inst.strategy_name} — {result['reason']}")
    if db_url:
        _record_result(db_url, inst.strategy_name, result["passed"])
    return result


def run_paper_arena(
    instances: list[PaperInstance],
    started_at: float,
    db_url: str | None = None,
) -> dict[str, dict]:
    """Poll every instance hourly until each run has ended, then finish it.

    Each run ends on its own (see ``end_reason``), and is judged only on the
    metrics from its last poll. Returns the comparison result per strategy.
    """
    running = list(instances)
    results: dict[str, dict] = {}
    while True:
        metrics = {m["strategy"]: m for m in _poll(running, db_url)}
        now = time.time()
        for inst in list(running):
            m = metrics.get(inst.strategy_name)
            reason = end_reason(m, started_at, now)
            if reason is None:
                continue
            print(f"Paper run for {inst.strategy_name} ended: {reason}")
            running.remove(inst)
            results[inst.strategy_name] = finish_run(db_url, inst, m, started_at, now)
        if not running:
            return results
        time.sleep(POLL_INTERVAL_SECS)


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


def run_cohort(db_url: str, names: list[str]) -> None:
    """Paper-trade one cohort until every run has ended and record each outcome."""
    instances = [spawn_paper_instance(name, slot) for slot, name in enumerate(names)]
    _mark_started(db_url, names)
    try:
        for inst in instances:
            if not wait_until_ready(inst):
                print(f"  Warning: {inst.container_name} did not answer /api/v1/ping in time")
        run_paper_arena(instances, started_at=time.time(), db_url=db_url)
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
