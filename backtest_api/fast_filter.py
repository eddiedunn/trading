"""Phase 1 fast filter — numpy/pandas signal quality metrics.

Evaluates agent-written signal functions against OHLCV data.
Sub-millisecond per evaluation, 10,000 variants in under 10 seconds.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

TAKER_FEE = 0.00045  # Hyperliquid taker rate (conservative — maker is 0.00015)
FUNDING_RATE_PER_BAR = 0.0001  # ~0.01%/hr synthetic drag, applied per candle close

# Resolve paths relative to repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _REPO_ROOT / "data"
STRATEGIES_DIR = _REPO_ROOT / "strategies" / "candidates"

DEFAULT_PAIRS = ["BTC_USDC-USDC_4h", "ETH_USDC-USDC_4h", "SOL_USDC-USDC_4h"]


def run_fast_filter(
    strategy_name: str,
    pairs: list[str] | None = None,
    data_dir: Path | None = None,
    strategies_dir: Path | None = None,
) -> dict:
    """Load strategy module, run signal function on OHLCV data, compute metrics."""
    pairs = pairs or DEFAULT_PAIRS
    data_dir = data_dir or DATA_DIR
    strategies_dir = strategies_dir or STRATEGIES_DIR

    strategy_path = strategies_dir / f"{strategy_name}.py"
    spec = importlib.util.spec_from_file_location(strategy_name, str(strategy_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    results = {}
    for pair in pairs:
        feather_path = data_dir / f"{pair}.feather"
        if not feather_path.exists():
            continue
        df = pd.read_feather(feather_path)
        signals = mod.generate_signals(df)
        results[pair] = _compute_metrics(df["close"], signals)

    if not results:
        return {"error": "No data files found", "per_pair": {}}

    return _aggregate_metrics(results)


def _compute_metrics(close: pd.Series, signals: pd.Series) -> dict:
    """Compute signal quality metrics from a position series."""
    # Returns per bar (shifted to avoid lookahead)
    pos = signals.shift(1).fillna(0)
    returns = pos * close.pct_change()

    # Apply round-trip fee on each position change
    trades_mask = pos.diff().abs() > 0
    returns[trades_mask] -= TAKER_FEE

    # Apply synthetic funding drag on held positions
    # KNOWN LIMITATION: Freqtrade Phase 2 does not model Hyperliquid hourly funding
    held_mask = pos.abs() > 0
    returns[held_mask] -= FUNDING_RATE_PER_BAR

    # Cumulative equity curve
    equity = (1 + returns).cumprod()

    # Sharpe (annualized for 4h bars — 6 bars/day × 365 days)
    bars_per_year = 6 * 365
    sharpe = (
        (returns.mean() / returns.std()) * np.sqrt(bars_per_year)
        if returns.std() > 0
        else 0
    )

    # Max drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_drawdown = drawdown.min()

    # Trade segmentation (each continuous position is a trade)
    trade_boundaries = pos.diff().fillna(0).abs() > 0
    trade_ids = trade_boundaries.cumsum() * (pos != 0)
    trade_returns = returns.groupby(trade_ids).sum()
    trade_returns = trade_returns[trade_returns.index > 0]  # drop flat periods

    win_count = (trade_returns > 0).sum()
    loss_count = (trade_returns <= 0).sum()
    trade_count = len(trade_returns)
    win_rate = win_count / trade_count if trade_count > 0 else 0

    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = abs(trade_returns[trade_returns <= 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_return = equity.iloc[-1] - 1 if len(equity) > 0 else 0

    # Calmar (annualized return / max drawdown)
    years = len(close) / bars_per_year
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        "total_return": round(float(total_return), 4),
        "sharpe": round(float(sharpe), 4),
        "max_drawdown": round(float(max_drawdown), 4),
        "win_rate": round(float(win_rate), 4),
        "profit_factor": round(float(profit_factor), 4),
        "trade_count": int(trade_count),
        "calmar": round(float(calmar), 4),
    }


def _aggregate_metrics(results: dict) -> dict:
    """Average metrics across pairs."""
    keys = ["total_return", "sharpe", "max_drawdown", "win_rate", "profit_factor", "calmar"]
    agg = {}
    for k in keys:
        vals = [
            r[k]
            for r in results.values()
            if isinstance(r[k], (int, float)) and r[k] != float("inf")
        ]
        agg[k] = round(np.mean(vals), 4) if vals else 0
    agg["trade_count"] = sum(r["trade_count"] for r in results.values())
    agg["per_pair"] = results
    return agg


def meets_phase1_criteria(stats: dict) -> bool:
    """Phase 1 gate: PF>1.3, DD>-20%, Sharpe>0.8, WR>45%, 30+ trades, return>20%."""
    return (
        stats.get("total_return", 0) > 0.20
        and stats.get("max_drawdown", -1) > -0.20
        and stats.get("profit_factor", 0) > 1.30
        and stats.get("win_rate", 0) > 0.45
        and stats.get("trade_count", 0) > 30
        and stats.get("sharpe", 0) > 0.80
    )
