"""Phase 1 fast filter — numpy/pandas signal quality metrics.

Evaluates agent-written signal functions against OHLCV data.
Sub-millisecond per evaluation, 10,000 variants in under 10 seconds.
"""

import importlib.util
import math
import os
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from backtest_api import periods
from backtest_api.benchmark import BARS_PER_YEAR, alpha_beta, buy_and_hold, portfolio_returns, sharpe

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

# Coins whose close and funding every pair's df carries as close_<COIN> / funding_<COIN>,
# whatever pairs the run itself covers, so a strategy always sees the same columns.
CONTEXT_COINS = ("BTC", "ETH", "SOL")


class SignalError(ValueError):
    """generate_signals returned something outside the documented contract."""


def run_fast_filter(
    strategy_name: str,
    pairs: list[str] | None = None,
    data_dir: Path | None = None,
    strategies_dir: Path | None = None,
    attempts: int = 1,
) -> dict:
    """Load a strategy module, run its signals on each pair's development bars, score it.

    Only bars before ``periods.holdout_start()`` are ever passed to
    ``generate_signals`` (and to the look-ahead check), so Phase 1 never sees
    the held-back final-test data.

    The headline metrics are for one combined account: an equal-weight
    portfolio of the pairs, rebalanced every bar (see ``_portfolio_stats``).
    ``per_pair`` keeps each pair's own metrics. ``attempts`` only sets the
    Sharpe bar shown in ``stats["gate"]``; ``meets_phase1_criteria`` recomputes
    the gate with the attempt count its caller passes.
    """
    pairs = pairs or DEFAULT_PAIRS
    data_dir = data_dir or DATA_DIR
    strategies_dir = strategies_dir or STRATEGIES_DIR

    strategy_path = strategies_dir / f"{strategy_name}.py"
    spec = importlib.util.spec_from_file_location(strategy_name, str(strategy_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    results = {}
    net_returns: dict[str, pd.Series] = {}
    trades: list[pd.Series] = []
    closes: dict[str, pd.Series] = {}
    lookahead_reasons = []
    for pair in pairs:
        feather_path = data_dir / f"{pair}.feather"
        if not feather_path.exists():
            continue
        df = periods.development_part(pd.read_feather(feather_path))
        if df.empty:
            continue
        funding = load_funding(data_dir, pair, df["timestamp"])
        df = add_context_columns(df, pair, data_dir, funding)
        try:
            signals = validate_signals(mod.generate_signals(df.copy()), df.index)
            lookahead = _lookahead_mismatch(mod.generate_signals, df, signals)
        except SignalError as e:
            return {"error": f"{pair}: {e}", "invalid_signals": True, "per_pair": {}}

        timestamps = df["timestamp"]
        returns, trade_returns = _simulate(df["close"], signals, funding=funding)
        metrics = _metrics(returns, trade_returns, timestamps=timestamps)
        if funding is None:
            metrics["funding"] = "assumed"
        else:
            metrics["funding"] = "real" if funding.notna().all() else "partial"
        metrics["lookahead"] = lookahead is not None
        if lookahead is not None:
            lookahead_reasons.append(f"{pair}: {lookahead}")
        results[pair] = metrics

        ts_index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
        net_returns[pair] = pd.Series(returns.to_numpy(), index=ts_index)
        closes[pair] = pd.Series(df["close"].to_numpy(dtype=float), index=ts_index)
        trades.append(trade_returns)

    if not results:
        return {"error": "No data files found", "per_pair": {}}

    portfolio = portfolio_net_returns(net_returns)
    stats = _portfolio_stats(portfolio, pd.concat(trades) if trades else pd.Series(dtype=float))
    stats["per_pair"] = results
    stats["floor_failures"] = floor_failures(results)
    stats["benchmark"] = buy_and_hold(closes)
    stats["alpha_beta"] = alpha_beta(portfolio, portfolio_returns(closes))
    stats["development_end"] = str(portfolio.index[-1]) if len(portfolio) else None
    stats["lookahead"] = bool(lookahead_reasons)
    if lookahead_reasons:
        stats["rejected_reason"] = (
            "Look-ahead: signals change when future bars are removed, so generate_signals "
            "reads data from after the bar it is deciding on. " + "; ".join(lookahead_reasons)
        )
    stats["gate"] = phase1_gate(stats, attempts)
    return stats


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


def _coin(pair: str) -> str:
    """'BTC_USDC-USDC_4h' -> 'BTC'."""
    return pair.split("_", 1)[0]


def _timeframe(pair: str) -> str:
    """'BTC_USDC-USDC_4h' -> '4h'."""
    return pair.rsplit("_", 1)[1]


def add_context_columns(
    df: pd.DataFrame, pair: str, data_dir: Path, funding: pd.Series | None = None
) -> pd.DataFrame:
    """Return ``df`` with the extra columns ``generate_signals`` gets. Every value is known
    at the close of its bar, so a signal computed from row t uses nothing after bar t.

    - ``funding_rate``: this pair's signed funding summed over the bar's own hours, i.e. the
      payments stamped T+1h .. T+len for the bar opening at T (see ``load_funding``). The last
      of those is paid at the bar's close, so the whole sum is known when the bar closes. It is
      the same per-bar funding a position held over this bar pays in ``_simulate``. NaN where
      there is no funding data (file missing, or the bar is outside the file's range).
    - ``close_<COIN>`` for BTC, ETH, SOL: that coin's close for the bar with the same open
      time (the same moment). NaN where that coin has no bar.
    - ``funding_<COIN>``: that coin's ``funding_rate``, aligned the same way.

    ``funding`` is this pair's already-loaded ``load_funding`` result (None if no file).
    Rows are matched by timestamp, so any prefix of the result equals the result computed on
    that prefix: the look-ahead prefix check still applies unchanged.
    """
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    own = funding if funding is not None else load_funding(data_dir, pair, out["timestamp"])
    out["funding_rate"] = np.nan if own is None else np.asarray(own, dtype=float)
    tf = _timeframe(pair)
    for coin in CONTEXT_COINS:
        other = f"{coin}_USDC-USDC_{tf}"
        if other == pair:
            out[f"close_{coin}"] = out["close"].to_numpy(dtype=float)
            out[f"funding_{coin}"] = out["funding_rate"].to_numpy(dtype=float)
            continue
        close = pd.Series(np.nan, index=out.index)
        path = data_dir / f"{other}.feather"
        if path.exists():
            o = pd.read_feather(path, columns=["timestamp", "close"])
            by_ts = pd.Series(o["close"].to_numpy(dtype=float),
                              index=pd.DatetimeIndex(pd.to_datetime(o["timestamp"], utc=True)))
            by_ts = by_ts[~by_ts.index.duplicated(keep="last")]
            close = pd.Series(by_ts.reindex(pd.DatetimeIndex(ts)).to_numpy(), index=out.index)
        out[f"close_{coin}"] = close.to_numpy(dtype=float)
        f = load_funding(data_dir, other, out["timestamp"])
        out[f"funding_{coin}"] = np.nan if f is None else np.asarray(f, dtype=float)
    return out


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
    """Metrics for one pair's position series (see ``_simulate`` for the cost model)."""
    returns, trade_returns = _simulate(close, signals, funding=funding)
    return _metrics(returns, trade_returns, timestamps=timestamps)


def _simulate(
    close: pd.Series,
    signals: pd.Series,
    funding: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Per-bar net returns and per-trade returns for one pair.

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
    return returns, trade_returns


def _years(n_bars: int, timestamps=None) -> float:
    if timestamps is not None and len(timestamps) > 1:
        ts = pd.to_datetime(pd.Series(timestamps), utc=True)
        return float((ts.iloc[-1] - ts.iloc[0]) / pd.Timedelta(days=365.25))
    return n_bars / BARS_PER_YEAR


def _metrics(returns: pd.Series, trade_returns: pd.Series, timestamps=None) -> dict:
    """Equity-curve metrics from per-bar returns, trade metrics from per-trade returns."""
    equity = (1 + returns).cumprod()
    max_drawdown = float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0
    total_return = float(equity.iloc[-1] - 1) if len(equity) else 0.0

    cagr = _cagr(total_return, _years(len(returns), timestamps))
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

    trade_count = len(trade_returns)
    win_rate = (trade_returns > 0).sum() / trade_count if trade_count > 0 else 0
    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = abs(trade_returns[trade_returns <= 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_return": round(float(total_return), 4),
        "sharpe": round(sharpe(returns), 4),
        "max_drawdown": round(float(max_drawdown), 4),
        "win_rate": round(float(win_rate), 4),
        "profit_factor": round(float(profit_factor), 4),
        "trade_count": int(trade_count),
        "calmar": round(float(calmar), 4),
    }


def portfolio_net_returns(pair_returns: dict[str, pd.Series]) -> pd.Series:
    """Per-bar net return of one account split equally across the pairs.

    Each series is a pair's per-bar net return indexed by bar timestamp. The
    account's return on a bar is the mean of the pairs' returns on that bar,
    which is an equal-weight portfolio rebalanced back to 1/N every bar (no
    rebalancing fees are charged). A bar where some pair has no data averages
    over the pairs that do, the same convention as ``benchmark.portfolio_returns``.
    """
    frame = pd.DataFrame(pair_returns).sort_index()
    return frame.mean(axis=1, skipna=True).fillna(0.0)


def _portfolio_stats(portfolio: pd.Series, pooled_trades: pd.Series) -> dict:
    """Headline stats: equity metrics on the combined account, trade metrics pooled."""
    m = _metrics(portfolio, pooled_trades, timestamps=portfolio.index)
    years = _years(len(portfolio), portfolio.index)
    total_return = float((1 + portfolio).prod() - 1) if len(portfolio) else 0.0
    return {
        "total_return": m["total_return"],
        "cagr": round(float(_cagr(total_return, years)), 4),
        "sharpe": m["sharpe"],
        "max_drawdown": m["max_drawdown"],
        "calmar": m["calmar"],
        "profit_factor": m["profit_factor"],
        "win_rate": m["win_rate"],
        "trade_count": m["trade_count"],
        "years": round(years, 4),
    }


def _cagr(total_return: float, years: float) -> float:
    if years <= 0:
        return 0.0
    if total_return <= -1:
        return -1.0  # wiped out; avoid a complex root
    return (1 + total_return) ** (1 / years) - 1


# Per-pair floor: no single pair may carry the rest.
FLOOR_MIN_PROFIT_FACTOR = 1.0
FLOOR_MIN_MAX_DRAWDOWN = -0.35


def floor_failures(per_pair: dict) -> list[dict]:
    """Pairs that break the floor: profit_factor >= 1.0 and max_drawdown > -0.35 each."""
    out = []
    for pair, m in per_pair.items():
        if m["profit_factor"] < FLOOR_MIN_PROFIT_FACTOR:
            out.append({"pair": pair, "check": "profit_factor", "value": m["profit_factor"],
                        "threshold": FLOOR_MIN_PROFIT_FACTOR})
        if not m["max_drawdown"] > FLOOR_MIN_MAX_DRAWDOWN:
            out.append({"pair": pair, "check": "max_drawdown", "value": m["max_drawdown"],
                        "threshold": FLOOR_MIN_MAX_DRAWDOWN})
    return out


MIN_SHARPE = 0.8
_EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(attempts: float, years: float) -> float:
    """Annualised Sharpe the best of ``attempts`` zero-edge strategies is expected to show.

    Uses the expected maximum of N independent standard normals,

        E[max Z_N] ~ (1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)),  g = Euler-Mascheroni,

    as in Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (J. Portfolio
    Management, 2014), eq. for SR_0 with V[SR] set to the zero-edge sampling
    variance. A zero-edge strategy's annualised Sharpe estimated over ``years``
    has standard error ~ 1/sqrt(years), so the noise bar is E[max Z_N]/sqrt(years).
    Returns 0 for attempts <= 1. Non-decreasing in attempts, decreasing in years.
    ``attempts`` may be fractional (revisions count as part of a trial); the
    approximation dips below zero just above N=1, so it is floored at 0.
    """
    if attempts <= 1:
        return 0.0
    if years <= 0:
        return float("inf")
    n = float(attempts)
    inv = NormalDist().inv_cdf
    e_max = (1 - _EULER_GAMMA) * inv(1 - 1 / n) + _EULER_GAMMA * inv(1 - 1 / (n * math.e))
    return max(0.0, e_max) / math.sqrt(years)


def required_sharpe(attempts: int, years: float, benchmark_sharpe: float) -> float:
    """Sharpe a strategy must beat: max(0.8, buy-and-hold Sharpe) + the multiple-testing noise bar."""
    return max(MIN_SHARPE, benchmark_sharpe) + expected_max_sharpe(attempts, years)


# Phase 1 gate thresholds (portfolio-level unless noted).
GATE_MIN_CAGR = 0.10
GATE_MIN_MAX_DRAWDOWN = -0.20
GATE_MIN_PROFIT_FACTOR = 1.3  # pooled trades
GATE_MIN_TRADES = 30


def phase1_gate(stats: dict, attempts: int = 1) -> dict:
    """Each Phase 1 check as ``{name: {value, threshold, passed}}``.

    Every check except ``lookahead`` and ``floor`` requires value > threshold.
    """
    bench_sharpe = (stats.get("benchmark") or {}).get("sharpe", 0.0)
    years = stats.get("years", 0.0)
    req = required_sharpe(attempts, years, bench_sharpe)

    def above(name, default, threshold):
        v = stats.get(name, default)
        return {"value": v, "threshold": threshold, "passed": bool(v > threshold)}

    failures = stats.get("floor_failures")
    gate = {
        "cagr": above("cagr", 0.0, GATE_MIN_CAGR),
        "max_drawdown": above("max_drawdown", -1.0, GATE_MIN_MAX_DRAWDOWN),
        "profit_factor": above("profit_factor", 0.0, GATE_MIN_PROFIT_FACTOR),
        "trade_count": above("trade_count", 0, GATE_MIN_TRADES),
        "sharpe": {**above("sharpe", 0.0, round(req, 4)), "required_sharpe": round(req, 4),
                   "attempts": attempts, "years": years, "benchmark_sharpe": bench_sharpe},
        "floor": {"value": failures if failures is not None else "not computed", "threshold": [],
                  "passed": failures == []},
        "lookahead": {"value": bool(stats.get("lookahead", False)), "threshold": False,
                      "passed": not stats.get("lookahead", False)},
    }
    return gate


def meets_phase1_criteria(stats: dict, attempts: int = 1) -> bool:
    """Phase 1 gate on the combined account. All must hold:

    CAGR > 10%, max drawdown > -20%, pooled profit factor > 1.3, more than 30
    trades, Sharpe > required_sharpe(attempts, years, buy-and-hold Sharpe), no
    pair below the per-pair floor, and no look-ahead.

    Recomputes ``stats["gate"]`` with this ``attempts`` (so the stored breakdown
    matches the verdict). Win rate is reported but not gated: trend and breakout
    strategies win under half their trades and still pay, and profit factor
    already covers it.
    """
    gate = phase1_gate(stats, attempts)
    stats["gate"] = gate
    return all(check["passed"] for check in gate.values())
