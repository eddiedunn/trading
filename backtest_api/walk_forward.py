"""Phase 2 — Freqtrade walk-forward validation.

Runs strategy through 3 time windows (in-sample, validation, out-of-sample)
via Freqtrade backtesting in a podman container. Only strategies that pass
Phase 1 should reach here.
"""

import json
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

WINDOWS = [
    ("20230101", "20240601", "in-sample"),
    ("20240601", "20250601", "validation"),
    ("20250601", "20260101", "out-of-sample"),
]

# Minimum thresholds per window
MIN_PROFIT_FACTOR = 1.2
MAX_DRAWDOWN = -0.25


def run_freqtrade_backtest(
    strategy_name: str,
    timerange: str,
    config_path: str = "/freqtrade/config/backtest.json",
) -> dict | None:
    """Run a single Freqtrade backtest via podman and return parsed results."""
    strategies_dir = str(_REPO_ROOT / "strategies")
    data_dir = str(_REPO_ROOT / "data")
    results_dir = str(_REPO_ROOT / "backtest_results")
    config_dir = str(_REPO_ROOT / "config")

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    export_filename = f"/freqtrade/user_data/backtest_results/{strategy_name}.json"

    result = subprocess.run(
        [
            "podman", "run", "--rm",
            "-v", f"{strategies_dir}:/freqtrade/strategies:ro,Z",
            "-v", f"{data_dir}:/freqtrade/user_data/data:ro,Z",
            "-v", f"{results_dir}:/freqtrade/user_data/backtest_results:Z",
            "-v", f"{config_dir}:/freqtrade/config:ro,Z",
            "freqtradeorg/freqtrade:stable",
            "backtesting",
            "--config", config_path,
            "--strategy", strategy_name,
            "--timerange", timerange,
            "--export", "trades",
            "--export-filename", export_filename,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        return {"error": result.stderr, "returncode": result.returncode}

    results_file = Path(results_dir) / f"{strategy_name}.json"
    if results_file.exists():
        return json.loads(results_file.read_text())
    return None


def walk_forward_test(
    strategy_name: str,
    timerange: str | None = None,
) -> dict:
    """Run walk-forward validation across 3 windows. Returns pass/fail + per-window stats."""
    windows_results = []
    passed = True

    for start, end, label in WINDOWS:
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
            return strategy_data.get("profit_factor", 0)
        # Fallback: direct key
        return stats.get("profit_factor", 0)
    except (AttributeError, TypeError):
        return 0


def _extract_max_drawdown(stats: dict) -> float:
    """Extract max drawdown from Freqtrade backtest results JSON."""
    try:
        for strategy_data in stats.get("strategy", {}).values():
            return strategy_data.get("max_drawdown", -1)
        return stats.get("max_drawdown", -1)
    except (AttributeError, TypeError):
        return -1
