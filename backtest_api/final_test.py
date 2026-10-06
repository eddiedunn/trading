"""The final test: one Freqtrade backtest on the held-back data.

Runs from holdout_start() to the latest candle every pair has, reusing the
Phase 2 runner and gates. Freqtrade loads its warm-up candles from before
holdout_start(); they feed indicators only and are not scored.

This module only measures. The API decides who may run it and records that
each strategy gets it once.

Sharpe comparison. Freqtrade's reported ``sharpe`` divides the mean daily
profit by the standard deviation of per-trade profits, so it can't be compared
with a buy-and-hold Sharpe. Both sides are therefore computed here from daily
returns, annualised with sqrt(365):
  * strategy: Freqtrade's ``daily_profit`` (closed-trade profit per day),
    filled with 0 on days with no closes, divided by the balance at the start
    of the day;
  * benchmark: the equal-weight portfolio of the 4h feathers' holdout_part,
    sampled at each day's last close.
The reported benchmark also includes benchmark.buy_and_hold on the 4h bars
(total return, 4h Sharpe, drawdown) for reference.
"""

import numpy as np
import pandas as pd

from backtest_api import walk_forward as wf
from backtest_api.benchmark import buy_and_hold, portfolio_returns
from backtest_api.periods import holdout_part

MIN_TRADES = 15
DAYS_PER_YEAR = 365


def annualised_sharpe(returns: pd.Series, periods_per_year: int = DAYS_PER_YEAR) -> float:
    std = returns.std()
    if len(returns) < 2 or not std > 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def strategy_daily_returns(block: dict, first_day, last_day) -> pd.Series:
    """Daily return on the balance from Freqtrade's daily_profit [[date, profit_abs], ...]."""
    def day(value) -> pd.Timestamp:  # daily_profit dates are naive UTC days
        ts = pd.Timestamp(value)
        return (ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts).normalize()

    days = pd.date_range(day(first_day), day(last_day), freq="D")
    pnl = pd.Series({day(d): float(p) for d, p in block.get("daily_profit") or []}, dtype=float)
    pnl = pnl.reindex(days, fill_value=0.0)
    start_balance = float(block.get("starting_balance") or block.get("dry_run_wallet") or 0.0)
    if start_balance <= 0:
        return pd.Series(dtype=float)
    balance_before = start_balance + pnl.cumsum().shift(1, fill_value=0.0)
    return pnl / balance_before


def holdout_closes(last_candle) -> dict[str, pd.DataFrame]:
    """Each pair's holdout_part (timestamp, close), cut at the last scored candle."""
    out = {}
    for pair in wf.PAIRS:
        df = holdout_part(pd.read_feather(wf.data_dir() / f"{pair}.feather"))
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df = df[ts <= pd.Timestamp(last_candle)]
        out[pair] = pd.Series(df["close"].to_numpy(), index=pd.to_datetime(df["timestamp"], utc=True))
    return out


def benchmark_stats(last_candle) -> dict:
    closes = holdout_closes(last_candle)
    stats = buy_and_hold(closes)
    daily = {pair: c.resample("1D").last().dropna() for pair, c in closes.items()}
    stats["sharpe_daily"] = round(annualised_sharpe(portfolio_returns(daily).iloc[1:]), 4)
    return stats


def run_final_test(strategy_name: str, final_tests: int = 1) -> dict:
    """One backtest on the holdout; returns
    {"passed", "window", "stats", "benchmark", "gate": {check: {value, threshold, passed}}, "required_sharpe"}.

    The Sharpe bar is buy-and-hold's daily Sharpe over the holdout plus the
    expected best Sharpe of ``final_tests`` zero-edge strategies over the holdout's
    length (fast_filter.expected_max_sharpe). Phase 1 is only a screen, so this is
    where the campaign pays for trying many strategies.
    """
    start = wf._holdout_start_dt()
    _, last_candle = wf.common_candle_range()
    end = last_candle + wf.CANDLE
    window = {"start": start.isoformat(), "end": end.isoformat(), "last_candle": last_candle.isoformat()}
    if end <= start:
        return {"passed": False, "window": window, "stats": None, "benchmark": None, "gate": {},
                "error": f"No holdout data: last common candle {last_candle} is before {start}"}

    result = wf.run_window(strategy_name, start, end, MIN_TRADES)
    stats = result.pop("stats", None)
    gate = result.pop("gate", {})
    error = result.pop("error", None)
    window.update(timerange=result["timerange"], backtest_start=result.get("backtest_start"),
                  backtest_end=result.get("backtest_end"))
    benchmark = benchmark_stats(last_candle)
    from backtest_api.fast_filter import expected_max_sharpe

    holdout_years = max((last_candle - start).total_seconds(), 0) / (365 * 86400)
    required = round(benchmark["sharpe_daily"] + expected_max_sharpe(final_tests, holdout_years), 4)

    if gate:
        block = wf.strategy_block(stats)
        daily = strategy_daily_returns(block, start, last_candle)
        strat_sharpe = round(annualised_sharpe(daily), 4)
        gate["sharpe_vs_buy_and_hold"] = wf._check(strat_sharpe, required, strat_sharpe > required)
        summary = {
            "trades": gate["total_trades"]["value"],
            "profit_total": gate["profit_total"]["value"],
            "profit_factor": gate["profit_factor"]["value"],
            "max_drawdown": gate["max_drawdown"]["value"],
            "sharpe_daily": strat_sharpe,
            "freqtrade_sharpe": block.get("sharpe"),
        }
    else:
        summary = None

    out = {
        "passed": bool(gate) and error is None and all(g["passed"] for g in gate.values()),
        "window": window,
        "stats": summary,
        "benchmark": benchmark,
        "gate": gate,
        "final_tests": final_tests,
        "required_sharpe": required,
    }
    if error:
        out["error"] = error
    return out
