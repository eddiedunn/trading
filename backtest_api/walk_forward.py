"""Phase 2 — Freqtrade walk-forward validation.

Runs a strategy through 3 equal, consecutive periods of the development data
(before holdout_start()) via Freqtrade in a podman container. Nothing is fitted,
so the periods are just "period 1/2/3". Only strategies that pass
Phase 1 should reach here.

Paths come from env so the same code runs on a laptop and inside the
backtest-api container. In the container, every directory is bind-mounted at
the same path it has on the host, so the paths handed to the sibling
Freqtrade container (spawned through the podman socket) resolve on the host.
"""

import json
import os
import subprocess
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest_api import universe
from backtest_api.periods import holdout_start

_REPO_ROOT = Path(__file__).resolve().parent.parent

FREQTRADE_IMAGE = "docker.io/freqtradeorg/freqtrade:stable"

# Gates, checked in every Phase 2 period.
MIN_PROFIT_FACTOR = 1.2
MAX_DRAWDOWN = -0.25
MIN_TRADES = 10

CANDLE = timedelta(hours=4)
N_PERIODS = 3
# Skip this much data at the start so Freqtrade can load indicator warm-up
# candles before period 1. If a strategy needs more, Freqtrade moves the start
# forward and the range check below fails the period.
WARMUP = timedelta(days=20)

PAIRS = universe.PAIRS


def _path(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, str(default)))


def freqtrade_data_dir() -> Path:
    return _path("TRADING_FT_DATA_DIR", _REPO_ROOT / "data" / "freqtrade")


def strategies_dir() -> Path:
    return _path("TRADING_STRATEGIES_DIR", _REPO_ROOT / "strategies" / "candidates")


def results_dir() -> Path:
    return _path("TRADING_RESULTS_DIR", _REPO_ROOT / "backtest_results")


def config_dir() -> Path:
    return _path("TRADING_CONFIG_DIR", _REPO_ROOT / "config")


def data_dir() -> Path:
    """The Phase 1 4h feathers (``<PAIR>_4h.feather`` with a ``timestamp`` column)."""
    return _path("TRADING_DATA_DIR", _REPO_ROOT / "data")


def common_candle_range(pairs: list[str] | None = None) -> tuple[datetime, datetime]:
    """(first, last) 4h candle open time that every core pair has, in UTC.

    Anchored on ``universe.CORE_PAIRS`` so the scoring periods keep the full BTC/ETH/SOL
    history; pairs listed later contribute the bars they have inside each period.
    """
    firsts, lasts = [], []
    for pair in pairs or universe.CORE_PAIRS:
        path = data_dir() / f"{pair}.feather"
        ts = pd.to_datetime(pd.read_feather(path, columns=["timestamp"])["timestamp"], utc=True)
        firsts.append(ts.min())
        lasts.append(ts.max())
    return max(firsts).to_pydatetime(), min(lasts).to_pydatetime()


def _holdout_start_dt() -> datetime:
    return datetime.combine(holdout_start(), datetime.min.time(), tzinfo=timezone.utc)


def default_windows(first_candle: datetime | None = None) -> list[tuple[datetime, datetime, str]]:
    """Three equal, consecutive (start, end, label) periods covering the development data.

    ``end`` is exclusive: the period's last candle opens at ``end - CANDLE``.
    Period 1 starts WARMUP after the first common candle; period 3 ends at
    holdout_start(), so no period ever sees a holdout candle. Leftover candles
    that don't divide by three are dropped from the start.
    """
    first = first_candle or common_candle_range()[0]
    end = _holdout_start_dt()
    candles = (end - (first + WARMUP)) // CANDLE
    per = candles // N_PERIODS
    if per <= 0:
        raise ValueError(f"No development data between {first} + warm-up and {end}")
    start = end - N_PERIODS * per * CANDLE
    return [
        (start + i * per * CANDLE, start + (i + 1) * per * CANDLE, f"period {i + 1}")
        for i in range(N_PERIODS)
    ]


def ft_timerange(start: datetime, last_candle: datetime) -> str:
    """Freqtrade timerange in epoch seconds. Freqtrade keeps candles with
    start <= open time <= stop, so the stop is the last candle to include."""
    return f"{int(start.timestamp())}-{int(last_candle.timestamp())}"


def run_freqtrade_backtest(
    strategy_name: str,
    timerange: str,
    config_path: str = "/freqtrade/config/backtest.json",
) -> dict | None:
    """Run a single Freqtrade backtest via podman and return parsed results."""
    out_dir = results_dir() / strategy_name / timerange
    out_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "podman", "run", "--rm",
            # Run as the host user so result files are readable by the caller.
            "--userns=keep-id:uid=1000,gid=1000",
            "-v", f"{strategies_dir()}:/freqtrade/strategies:ro,Z",
            "-v", f"{freqtrade_data_dir()}:/freqtrade/user_data/data:ro,Z",
            "-v", f"{out_dir}:/freqtrade/user_data/backtest_results:Z",
            "-v", f"{config_dir()}:/freqtrade/config:ro,Z",
            FREQTRADE_IMAGE,
            "backtesting",
            "--config", config_path,
            "--strategy", strategy_name,
            "--strategy-path", "/freqtrade/strategies",
            "--timerange", timerange,
            "--export", "trades",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        return {"error": result.stderr, "returncode": result.returncode}

    return _load_latest_result(out_dir)


def _load_latest_result(out_dir: Path) -> dict | None:
    """Read the result Freqtrade names in .last_result.json (a zip holding the JSON)."""
    marker = out_dir / ".last_result.json"
    if not marker.exists():
        return None
    archive = out_dir / json.loads(marker.read_text())["latest_backtest"]
    if not archive.exists():
        return None
    with zipfile.ZipFile(archive) as zf:
        name = archive.with_suffix(".json").name
        return json.loads(zf.read(name))


def strategy_block(stats: dict) -> dict:
    """The per-strategy stats Freqtrade nests under ``strategy.<name>``."""
    try:
        for block in stats.get("strategy", {}).values():
            return block
        return stats
    except (AttributeError, TypeError):
        return {}


def _parse_ft_time(value) -> datetime | None:
    """Freqtrade writes backtest_start/end as naive UTC strings."""
    if not value:
        return None
    ts = pd.Timestamp(value)
    return (ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")).to_pydatetime()


def range_error(block: dict, start: datetime, last_candle: datetime) -> str | None:
    """Fail when Freqtrade actually tested a different range than asked (more than
    one candle off), e.g. missing data or a start moved for warm-up candles."""
    got_start = _parse_ft_time(block.get("backtest_start"))
    got_end = _parse_ft_time(block.get("backtest_end"))
    if got_start is None or got_end is None:
        return "Freqtrade result has no backtest_start/backtest_end"
    problems = []
    if abs(got_start - start) > CANDLE:
        problems.append(f"started {got_start:%Y-%m-%d %H:%M} instead of {start:%Y-%m-%d %H:%M}")
    if abs(got_end - last_candle) > CANDLE:
        problems.append(f"ended {got_end:%Y-%m-%d %H:%M} instead of {last_candle:%Y-%m-%d %H:%M}")
    if problems:
        return ("Freqtrade tested the wrong range (missing data, or warm-up moved the start): "
                + "; ".join(problems))
    return None


def _check(value, threshold, passed: bool) -> dict:
    return {"value": value, "threshold": threshold, "passed": bool(passed)}


def core_gates(block: dict, min_trades: int) -> dict:
    """The trade-count, profit factor, drawdown and profit gates shared by Phase 2
    and the final test, as {check: {value, threshold, passed}}.

    Freqtrade reports profit factor as 0/None when no trade lost. That is not a
    bad profit factor, so it passes when the trade and profit gates pass.
    """
    trades = int(block.get("total_trades") or 0)
    profit = float(block.get("profit_total") or 0.0)
    raw_pf = block.get("profit_factor")
    dd = _extract_max_drawdown({"strategy": {"s": block}})
    enough_trades = trades >= min_trades
    profitable = profit > 0
    if not raw_pf and enough_trades and profitable:
        pf_value, pf_passed = None, True  # no losing trades
    else:
        pf_value = round(float(raw_pf or 0.0), 4)
        pf_passed = pf_value >= MIN_PROFIT_FACTOR
    return {
        "total_trades": _check(trades, min_trades, enough_trades),
        "profit_factor": _check(pf_value, MIN_PROFIT_FACTOR, pf_passed),
        "max_drawdown": _check(round(dd, 4), MAX_DRAWDOWN, dd >= MAX_DRAWDOWN),
        "profit_total": _check(round(profit, 4), 0, profitable),
    }


def run_window(strategy_name: str, start: datetime, end: datetime, min_trades: int) -> dict:
    """Backtest [start, end) and apply the core gates. ``end`` is exclusive."""
    last_candle = end - CANDLE
    tr = ft_timerange(start, last_candle)
    window = {
        "timerange": tr,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "last_candle": last_candle.isoformat(),
    }
    stats = run_freqtrade_backtest(strategy_name, tr)
    if stats is None or "error" in stats:
        error = stats.get("error", "No results produced") if stats else "No results"
        return {**window, "passed": False, "error": error, "stats": stats}

    block = strategy_block(stats)
    gates = core_gates(block, min_trades)
    result = {
        **window,
        "backtest_start": block.get("backtest_start"),
        "backtest_end": block.get("backtest_end"),
        "trades": gates["total_trades"]["value"],
        "profit_total": gates["profit_total"]["value"],
        "profit_factor": gates["profit_factor"]["value"],
        "max_drawdown": gates["max_drawdown"]["value"],
        "gate": gates,
        "passed": all(g["passed"] for g in gates.values()),
        "stats": stats,
    }
    error = range_error(block, start, last_candle)
    if error:
        result.update(passed=False, error=error)
    return result


def walk_forward_test(
    strategy_name: str,
    timerange: str | None = None,
    windows: list[tuple[datetime, datetime, str]] | None = None,
) -> dict:
    """Run Phase 2 over 3 development periods. Returns pass/fail + per-period stats.

    ``timerange`` is accepted for API compatibility and ignored: the periods
    are fixed by the data and holdout_start().
    """
    results = []
    for start, end, label in windows or default_windows():
        result = run_window(strategy_name, start, end, MIN_TRADES)
        result.pop("stats", None)
        results.append({"label": label, **result})
    return {"passed": all(w["passed"] for w in results), "windows": results}


def _extract_profit_factor(stats: dict) -> float:
    """Extract profit factor from Freqtrade backtest results JSON."""
    try:
        # Freqtrade stores results under strategy name key
        for strategy_data in stats.get("strategy", {}).values():
            return strategy_data.get("profit_factor") or 0
        # Fallback: direct key
        return stats.get("profit_factor", 0)
    except (AttributeError, TypeError):
        return 0


def _extract_max_drawdown(stats: dict) -> float:
    """Extract max drawdown (as a negative fraction) from Freqtrade backtest results JSON.

    Current Freqtrade reports ``max_drawdown_account`` as a positive fraction;
    older results used a negative ``max_drawdown``.
    """
    try:
        for strategy_data in stats.get("strategy", {}).values():
            if strategy_data.get("max_drawdown_account") is not None:
                return -abs(strategy_data["max_drawdown_account"])
            return strategy_data.get("max_drawdown", -1)
        return stats.get("max_drawdown", -1)
    except (AttributeError, TypeError):
        return -1
