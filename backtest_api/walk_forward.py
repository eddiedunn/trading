"""Phase 2 — Freqtrade walk-forward validation.

Runs strategy through 3 time windows (in-sample, validation, out-of-sample)
via Freqtrade backtesting in a podman container. Only strategies that pass
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
from datetime import date, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

FREQTRADE_IMAGE = "docker.io/freqtradeorg/freqtrade:stable"

# Minimum thresholds per window
MIN_PROFIT_FACTOR = 1.2
MAX_DRAWDOWN = -0.25

# Hyperliquid only serves the last 5000 candles, and history is kept from
# 2023-12, so windows are anchored to today rather than fixed dates.
IN_SAMPLE_DAYS = 365
VALIDATION_DAYS = 182
OUT_OF_SAMPLE_DAYS = 182


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


def default_windows(today: date | None = None) -> list[tuple[str, str, str]]:
    """Return (start, end, label) windows ending today, oldest first."""
    end = today or date.today()
    oos_start = end - timedelta(days=OUT_OF_SAMPLE_DAYS)
    val_start = oos_start - timedelta(days=VALIDATION_DAYS)
    is_start = val_start - timedelta(days=IN_SAMPLE_DAYS)
    fmt = "%Y%m%d"
    return [
        (is_start.strftime(fmt), val_start.strftime(fmt), "in-sample"),
        (val_start.strftime(fmt), oos_start.strftime(fmt), "validation"),
        (oos_start.strftime(fmt), end.strftime(fmt), "out-of-sample"),
    ]


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


def walk_forward_test(
    strategy_name: str,
    timerange: str | None = None,
    windows: list[tuple[str, str, str]] | None = None,
) -> dict:
    """Run walk-forward validation across 3 windows. Returns pass/fail + per-window stats."""
    windows_results = []
    passed = True

    for start, end, label in windows or default_windows():
        stats = run_freqtrade_backtest(strategy_name, f"{start}-{end}")

        if stats is None or "error" in stats:
            windows_results.append({
                "label": label,
                "timerange": f"{start}-{end}",
                "passed": False,
                "error": stats.get("error", "No results produced") if stats else "No results",
            })
            passed = False
            continue

        # Extract metrics from Freqtrade results format
        pf = _extract_profit_factor(stats)
        dd = _extract_max_drawdown(stats)

        window_passed = pf >= MIN_PROFIT_FACTOR and dd >= MAX_DRAWDOWN
        if not window_passed:
            passed = False

        windows_results.append({
            "label": label,
            "timerange": f"{start}-{end}",
            "passed": window_passed,
            "profit_factor": round(pf, 4),
            "max_drawdown": round(dd, 4),
        })

    return {"passed": passed, "windows": windows_results}


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
