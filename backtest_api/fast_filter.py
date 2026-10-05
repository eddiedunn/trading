"""Phase 1 fast filter — numpy/pandas signal quality metrics.

Evaluates agent-written signal functions against OHLCV data.
Sub-millisecond per evaluation, 10,000 variants in under 10 seconds.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

TAKER_FEE = 0.00045  # Hyperliquid taker rate (conservative — maker is 0.00015)

# Used only for bars with no real funding data (pair's funding file missing, or
# bars outside its range). Hyperliquid's baseline is 0.00125%/hr = 0.005% per 4h
# bar. Charged on |pos|, so both longs and shorts pay it (conservative).
ASSUMED_FUNDING_PER_BAR = 0.00005

# Prefix cut points for the look-ahead check, as fractions of the data length.
LOOKAHEAD_CUTS = (0.5, 0.65, 0.8, 0.95)

# Resolve paths relative to repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("TRADING_DATA_DIR", str(_REPO_ROOT / "data")))
STRATEGIES_DIR = Path(os.environ.get("TRADING_STRATEGIES_DIR", str(_REPO_ROOT / "strategies" / "candidates")))

DEFAULT_PAIRS = ["BTC_USDC-USDC_4h", "ETH_USDC-USDC_4h", "SOL_USDC-USDC_4h"]


class SignalError(ValueError):
    """generate_signals returned something outside the documented contract."""


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
    lookahead_reasons = []
    for pair in pairs:
        feather_path = data_dir / f"{pair}.feather"
        if not feather_path.exists():
            continue
        df = pd.read_feather(feather_path)
        try:
            signals = validate_signals(mod.generate_signals(df.copy()), df.index)
            lookahead = _lookahead_mismatch(mod.generate_signals, df, signals)
        except SignalError as e:
            return {"error": f"{pair}: {e}", "invalid_signals": True, "per_pair": {}}

        timestamps = df["timestamp"] if "timestamp" in df.columns else None
        funding = load_funding(data_dir, pair, timestamps) if timestamps is not None else None
        metrics = _compute_metrics(df["close"], signals, funding=funding, timestamps=timestamps)
        if funding is None:
            metrics["funding"] = "assumed"
        else:
            metrics["funding"] = "real" if funding.notna().all() else "partial"
        metrics["lookahead"] = lookahead is not None
        if lookahead is not None:
            lookahead_reasons.append(f"{pair}: {lookahead}")
        results[pair] = metrics

    if not results:
        return {"error": "No data files found", "per_pair": {}}

    agg = _aggregate_metrics(results)
    agg["lookahead"] = bool(lookahead_reasons)
    if lookahead_reasons:
        agg["rejected_reason"] = (
            "Look-ahead: signals change when future bars are removed, so generate_signals "
            "reads data from after the bar it is deciding on. " + "; ".join(lookahead_reasons)
        )
    return agg


def validate_signals(raw, index: pd.Index) -> pd.Series:
    """Coerce generate_signals output to a float Series on ``index``; enforce {-1, 0, 1}."""
    if isinstance(raw, pd.DataFrame):
        if raw.shape[1] != 1:
            raise SignalError(f"generate_signals returned a DataFrame with {raw.shape[1]} columns, expected a Series")
        raw = raw.iloc[:, 0]
    if isinstance(raw, pd.Series):
        if len(raw) != len(index):
            raise SignalError(f"generate_signals returned {len(raw)} values for {len(index)} bars")
        if not raw.index.equals(index):
            if not (raw.index.is_unique and raw.index.isin(index).all()):
                raise SignalError("generate_signals returned a Series whose index does not match df.index")
            raw = raw.reindex(index)
    else:
        arr = np.asarray(raw)
        if arr.ndim != 1 or len(arr) != len(index):
            raise SignalError(f"generate_signals returned shape {arr.shape}, expected ({len(index)},)")
        raw = pd.Series(arr, index=index)
    try:
        s = pd.to_numeric(raw, errors="raise").astype(float)
    except (TypeError, ValueError) as e:
        raise SignalError(f"generate_signals returned non-numeric values: {e}") from None
    if s.isna().any():
        first = s.index[s.isna()][0]
        raise SignalError(f"generate_signals returned {int(s.isna().sum())} NaN values (first at index {first}); "
                          "fill warm-up bars with 0")
    bad = ~s.isin([-1.0, 0.0, 1.0])
    if bad.any():
        raise SignalError(f"generate_signals returned values outside {{-1, 0, 1}}: "
                          f"{sorted(set(s[bad].tolist()))[:5]}")
    return s


def _lookahead_mismatch(generate_signals, df: pd.DataFrame, full: pd.Series) -> str | None:
    """Recompute signals on prefixes of df; a causal strategy gives the same values.

    Returns a description of the first mismatch, or None. Invalid output on a
    prefix also counts: a causal strategy's prefix output equals the full output.
    """
    n = len(df)
    cuts = sorted({int(n * f) for f in LOOKAHEAD_CUTS if 1 <= int(n * f) < n})
    for k in cuts:
        prefix_df = df.iloc[:k].copy()
        try:
            prefix = validate_signals(generate_signals(prefix_df), prefix_df.index)
        except SignalError as e:
            return f"output on the first {k} bars is invalid ({e})"
        a = full.iloc[:k].to_numpy(dtype=float)
        b = prefix.to_numpy(dtype=float)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        if not same.all():
            i = int(np.argmax(~same))
            return f"bar {i} is {a[i]:g} with all data but {b[i]:g} with only the first {k} bars"
    return None


def funding_path(data_dir: Path, pair: str) -> Path:
    """'BTC_USDC-USDC_4h' -> data_dir/'BTC_USDC-USDC_funding_1h.feather' (written by scripts/download_data.py)."""
    stem = pair.rsplit("_", 1)[0]
    return data_dir / f"{stem}_funding_1h.feather"


def load_funding(data_dir: Path, pair: str, timestamps: pd.Series) -> pd.Series | None:
    """Sum hourly funding rates into each bar. None if the pair has no funding file.

    Timestamp convention: scripts/download_data.py floors Hyperliquid's funding
    time to the hour, and a payment stamped H is for the hour [H-1h, H). A
    position held over bar t (opened at the bar's open time T, held to T+len)
    pays the settlements at T+1h .. T+len, so a payment stamped H belongs to the
    bar whose open is <= H-1h < open+len.

    Bars outside the funding file's range are NaN (filled with the assumed rate
    in _compute_metrics). Sign: positive funding means longs pay, shorts receive.
    """
    path = funding_path(data_dir, pair)
    if not path.exists():
        return None
    fund = pd.read_feather(path)
    bar_ts = pd.to_datetime(timestamps, utc=True).reset_index(drop=True)
    out = pd.Series(np.nan, index=timestamps.index, dtype=float)
    if fund.empty or len(bar_ts) == 0:
        return out
    bar_len = bar_ts.diff().median() if len(bar_ts) > 1 else pd.Timedelta("4h")
    accrual_start = pd.to_datetime(fund["timestamp"], utc=True) - pd.Timedelta("1h")

    bar_ns = _ns(bar_ts)
    acc_ns = _ns(accrual_start)
    idx = np.searchsorted(bar_ns, acc_ns, side="right") - 1
    ok = (idx >= 0) & (acc_ns < bar_ns[np.clip(idx, 0, None)] + bar_len.value)
    sums = np.bincount(idx[ok], weights=fund["rate"].to_numpy(dtype=float)[ok], minlength=len(bar_ts))

    # Bars fully inside [first, last] funding hour count as covered (0 if no record).
    first, last = accrual_start.min(), accrual_start.max()
    covered = ((bar_ts >= first) & (bar_ts + bar_len - pd.Timedelta("1h") <= last)).to_numpy()
    out[:] = np.where(covered, sums, np.nan)
    return out


def _ns(ts: pd.Series) -> np.ndarray:
    """UTC timestamps as int64 nanoseconds, whatever unit the feather file used."""
    return ts.dt.tz_convert(None).to_numpy().astype("datetime64[ns]").astype("int64")


def _compute_metrics(
    close: pd.Series,
    signals: pd.Series,
    funding: pd.Series | None = None,
    timestamps: pd.Series | None = None,
) -> dict:
    """Compute signal quality metrics from a position series.

    ``funding`` is the signed funding rate summed over each bar (aligned to
    ``close``), positive = longs pay. NaN bars, or funding=None, use the assumed
    rate on |pos|.
    """
    signals = pd.Series(np.asarray(signals, dtype=float), index=close.index)
    # Position held over bar t was decided at the close of bar t-1 (no lookahead)
    pos = signals.shift(1).fillna(0.0)
    returns = pos * close.pct_change().fillna(0.0)

    # Taker fee on every unit of position changed: a long->short flip costs 2x.
    prev = pos.shift(1).fillna(0.0)
    fees = TAKER_FEE * (pos - prev).abs()
    returns -= fees

    # Funding (Freqtrade Phase 2 models real funding too).
    assumed = ASSUMED_FUNDING_PER_BAR * pos.abs()
    if funding is None:
        funding_cost = assumed
    else:
        funding = pd.Series(np.asarray(funding, dtype=float), index=close.index)
        funding_cost = (pos * funding).where(funding.notna(), assumed)
    returns -= funding_cost

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

    # Trade segmentation: a trade is a run of bars with the same nonzero side.
    side = np.sign(pos)
    prev_side = np.sign(prev)
    starts = (side != 0) & (side != prev_side)
    trade_ids = starts.cumsum() * (side != 0)
    prev_ids = trade_ids.shift(1).fillna(0).astype(int)

    # On a bar where the side changes, the part of the fee for closing the old
    # position belongs to the trade being closed (bar t-1's trade), not to bar t.
    side_changed = (side != prev_side) & (prev_side != 0)
    exit_fee = (TAKER_FEE * prev.abs()).where(side_changed, 0.0)
    bar_pnl = returns + exit_fee  # remove the exit part from bar t
    trade_returns = bar_pnl.groupby(trade_ids).sum().sub(
        exit_fee.groupby(prev_ids).sum(), fill_value=0.0
    )
    trade_returns = trade_returns[trade_returns.index > 0]  # drop flat periods

    win_count = (trade_returns > 0).sum()
    trade_count = len(trade_returns)
    win_rate = win_count / trade_count if trade_count > 0 else 0

    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = abs(trade_returns[trade_returns <= 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_return = equity.iloc[-1] - 1 if len(equity) > 0 else 0

    # Calmar (annualized return / max drawdown)
    if timestamps is not None and len(timestamps) > 1:
        ts = pd.to_datetime(timestamps, utc=True)
        years = (ts.iloc[-1] - ts.iloc[0]) / pd.Timedelta(days=365.25)
    else:
        years = len(close) / bars_per_year
    if years <= 0:
        annual_return = 0
    elif total_return <= -1:
        annual_return = -1.0  # wiped out; avoid a complex root
    else:
        annual_return = (1 + total_return) ** (1 / years) - 1
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
    """Phase 1 gate: PF>1.3, DD>-20%, Sharpe>0.8, 30+ trades, return>20%, no look-ahead.

    Win rate is reported but not gated: trend and breakout strategies win
    under half their trades and still pay, and profit factor already covers it.
    """
    return (
        not stats.get("lookahead", False)
        and stats.get("total_return", 0) > 0.20
        and stats.get("max_drawdown", -1) > -0.20
        and stats.get("profit_factor", 0) > 1.30
        and stats.get("trade_count", 0) > 30
        and stats.get("sharpe", 0) > 0.80
    )
