"""Compare a finished paper run with a backtest of the same strategy over the same weeks.

A paper run passes when it matches what the backtest predicts for its own
period, not when it clears an absolute profit bar. At the end of a run the
monitor:

1. downloads Hyperliquid candles and funding for the paper period plus a
   warm-up, and exports them in Freqtrade's futures layout (Freqtrade cannot
   download Hyperliquid history itself);
2. runs ``freqtrade backtesting`` in a sibling container with a config that
   matches the paper config, over exactly the paper period on 4h candles;
3. checks the paper results against that backtest (see ``TOLERANCES``).

If the backtest cannot run, the run fails with the reason.
"""

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd

from backtest_api.walk_forward import _load_latest_result
from paper.orchestrator import (
    FREQTRADE_IMAGE,
    PAIRS,
    build_comparison_config,
    paper_dir,
    strategies_dir,
)
from scripts.download_data import (
    REQUEST_INTERVAL_MS,
    download_funding,
    download_pair,
    export_freqtrade,
    freqtrade_pair_name,
    pair_to_filename,
)

TIMEFRAME = "4h"
CANDLE_SECS = 4 * 3600
# Candles before the paper start so indicators are warm when the backtest starts.
# Freqtrade loads the strategy's startup_candle_count before the timerange itself;
# 90 days is 540 4h candles.
WARMUP_DAYS = 90
BACKTEST_TIMEOUT_SECS = 1800

TOLERANCES = {
    # |paper - backtest| <= max(abs_trades, rel * backtest)
    "trade_count_rel": 0.40,
    "trade_count_abs": 3,
    # paper return >= backtest return - max(abs_pp, rel * |backtest return|)
    "return_abs_pp": 2.0,
    "return_rel": 0.50,
    # worst paper trade >= stoploss - this many percentage points
    "stop_slippage_pp": 1.0,
    # paper PF >= rel * backtest PF, checked only when the backtest has min_trades
    "profit_factor_rel": 0.80,
    "profit_factor_min_trades": 10,
}


def raw_data_dir() -> Path:
    """Downloaded candles and funding (download_data.py's own layout)."""
    return Path(os.environ.get("TRADING_PAPER_DATA_DIR", str(paper_dir() / "data")))


def freqtrade_data_dir() -> Path:
    """The same data exported in Freqtrade's futures layout."""
    return Path(os.environ.get("TRADING_PAPER_FT_DATA_DIR", str(paper_dir() / "ftdata")))


def results_dir() -> Path:
    return paper_dir() / "compare"


def candle_floor(ts: float) -> int:
    """Start of the 4h candle that contains ``ts`` (epoch seconds)."""
    return int(ts) // CANDLE_SECS * CANDLE_SECS


def prepare_data(start: float, end: float, pairs: list[str] = PAIRS) -> None:
    """Download candles and funding for [start - warm-up, end] and export them for Freqtrade.

    The 4h candle file is downloaded fresh each time: download_data dedups by
    keeping the existing row, so a candle that was still forming at the last
    download would otherwise never be corrected. Funding history is final once
    published, so it is extended incrementally.
    """
    since = (datetime.fromtimestamp(start, timezone.utc) - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    raw = raw_data_dir()
    raw.mkdir(parents=True, exist_ok=True)
    exchange = ccxt.hyperliquid({
        "options": {"defaultType": "swap"},
        "enableRateLimit": True,
        "rateLimit": REQUEST_INTERVAL_MS,
    })
    need_until = pd.Timestamp(candle_floor(end) - CANDLE_SECS, unit="s", tz="UTC")
    for pair in pairs:
        (raw / pair_to_filename(pair, TIMEFRAME)).unlink(missing_ok=True)
        download_pair(exchange, pair, TIMEFRAME, raw, since)
        download_funding(exchange, pair, raw, since)
        export_freqtrade(pair, raw, freqtrade_data_dir())
        out = freqtrade_data_dir() / "hyperliquid" / "futures" / f"{freqtrade_pair_name(pair)}-{TIMEFRAME}-futures.feather"
        last = pd.read_feather(out)["date"].max()
        if pd.isna(last) or last < need_until:
            raise RuntimeError(f"{pair} candles end at {last}, paper period needs {need_until}")


def run_comparison_backtest(strategy_name: str, start: float, end: float) -> dict:
    """Backtest the strategy over the paper period and return its Freqtrade strategy stats.

    Raises RuntimeError when Freqtrade fails or produces no result.
    """
    timerange = f"{candle_floor(start)}-{candle_floor(end)}"
    out_dir = results_dir() / strategy_name / timerange
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(build_comparison_config(), indent=2))

    # The monitor bind-mounts the paper dir at the same path it has on the host,
    # so these host paths are valid for the sibling container.
    result = subprocess.run(
        [
            "podman", "run", "--rm",
            "--userns=keep-id:uid=1000,gid=1000",
            "-v", f"{strategies_dir()}:/freqtrade/strategies:ro,Z",
            "-v", f"{freqtrade_data_dir()}:/freqtrade/user_data/data:ro,Z",
            "-v", f"{out_dir}:/freqtrade/user_data/backtest_results:Z",
            FREQTRADE_IMAGE,
            "backtesting",
            "--config", "/freqtrade/user_data/backtest_results/config.json",
            "--strategy", strategy_name,
            "--strategy-path", "/freqtrade/strategies",
            "--timeframe", TIMEFRAME,
            "--timerange", timerange,
            "--export", "trades",
        ],
        capture_output=True,
        text=True,
        timeout=BACKTEST_TIMEOUT_SECS,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-20:])
        raise RuntimeError(f"freqtrade backtesting exited {result.returncode}: {tail}")

    stats = _load_latest_result(out_dir)
    if not stats or strategy_name not in stats.get("strategy", {}):
        raise RuntimeError(f"freqtrade backtesting produced no result in {out_dir}")
    return stats["strategy"][strategy_name]


def profit_factor(profits_abs: list[float]) -> float:
    """Gross profit / gross loss. Infinite with wins and no losses, 0 with no wins."""
    wins = sum(p for p in profits_abs if p > 0)
    losses = -sum(p for p in profits_abs if p < 0)
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def summarize_backtest(bt: dict) -> dict:
    trades = bt.get("trades") or []
    return {
        "trade_count": bt.get("total_trades", len(trades)),
        "return_pct": (bt.get("profit_total") or 0) * 100,
        "profit_factor": profit_factor([t["profit_abs"] for t in trades]),
        "stoploss": bt.get("stoploss"),
    }


def summarize_paper(metrics: dict, trades: list[dict]) -> dict:
    """``trades`` holds closed and still-open trades, as the backtest force-exits open ones at its end."""
    ratios = [t["profit_ratio"] for t in trades if t.get("profit_ratio") is not None]
    return {
        "trade_count": metrics["trade_count"],
        "closed_trade_count": metrics.get("closed_trade_count"),
        "return_pct": metrics["profit_pct"],
        "profit_factor": profit_factor([t.get("profit_abs") or 0 for t in trades]),
        "worst_trade_pct": min(ratios) * 100 if ratios else None,
    }


def compare(paper: dict, backtest: dict, tol: dict = TOLERANCES) -> dict:
    """Check paper against backtest. Every check is ``{value, threshold, passed}``."""
    checks = {}

    bt_n, p_n = backtest["trade_count"], paper["trade_count"]
    slack = max(tol["trade_count_abs"], tol["trade_count_rel"] * bt_n)
    checks["trade_count"] = {
        "value": p_n,
        "threshold": [max(0.0, bt_n - slack), bt_n + slack],
        "passed": abs(p_n - bt_n) <= slack,
    }

    bt_ret = backtest["return_pct"]
    floor = bt_ret - max(tol["return_abs_pp"], tol["return_rel"] * abs(bt_ret))
    checks["return_pct"] = {
        "value": paper["return_pct"],
        "threshold": round(floor, 4),
        "passed": paper["return_pct"] >= floor,
    }

    stoploss = backtest.get("stoploss")
    worst = paper["worst_trade_pct"]
    if stoploss is None:
        checks["worst_trade_pct"] = {"value": worst, "threshold": None, "passed": False,
                                     "note": "backtest reported no stoploss"}
    else:
        limit = stoploss * 100 - tol["stop_slippage_pp"]
        checks["worst_trade_pct"] = {
            "value": worst,
            "threshold": round(limit, 4),
            "passed": worst is None or worst >= limit,
        }

    bt_pf, p_pf = backtest["profit_factor"], paper["profit_factor"]
    if bt_n >= tol["profit_factor_min_trades"]:
        need = round(tol["profit_factor_rel"] * bt_pf, 6)
        checks["profit_factor"] = {"value": p_pf, "threshold": need, "passed": p_pf >= need}
    else:
        checks["profit_factor"] = {
            "value": p_pf, "threshold": None, "passed": True,
            "note": f"not checked: backtest has fewer than {tol['profit_factor_min_trades']} trades",
        }

    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks}


def evaluate_paper_run(
    strategy_name: str, start: float, end: float, metrics: dict | None, trades: list[dict] | None
) -> dict:
    """Backtest the paper period and compare. Never raises: a failure is a fail with its reason."""
    result = {
        "strategy": strategy_name,
        "start": datetime.fromtimestamp(start, timezone.utc).isoformat(),
        "end": datetime.fromtimestamp(end, timezone.utc).isoformat(),
        "timerange": f"{candle_floor(start)}-{candle_floor(end)}",
        "tolerances": TOLERANCES,
    }
    if metrics is None or trades is None:
        return {**result, "passed": False, "reason": "no final paper metrics or trades"}
    paper = summarize_paper(metrics, trades)
    result["paper"] = paper
    try:
        prepare_data(start, end)
        backtest = summarize_backtest(run_comparison_backtest(strategy_name, start, end))
    except Exception as e:  # noqa: BLE001 - any failure to backtest is a recorded fail
        return {**result, "passed": False, "reason": f"comparison backtest could not run: {e}"}
    result["backtest"] = backtest
    outcome = compare(paper, backtest)
    failed = [k for k, c in outcome["checks"].items() if not c["passed"]]
    reason = "matches backtest" if outcome["passed"] else "failed: " + ", ".join(failed)
    return {**result, **outcome, "reason": reason}


def save_result(result: dict) -> Path:
    """Write the comparison next to the backtest output so it outlives the log."""
    path = results_dir() / result["strategy"] / f"comparison-{result['timerange']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str))
    return path
