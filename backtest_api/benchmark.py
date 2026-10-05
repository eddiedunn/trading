"""Buy-and-hold benchmark: an equal-weight, daily-rebalanced long of every pair.

Phase 1 and the final test compare a strategy against this over the same bars.
"""

import numpy as np
import pandas as pd

BARS_PER_YEAR = 6 * 365  # 4h bars, markets open every day


def portfolio_returns(closes: dict[str, pd.Series]) -> pd.Series:
    """Equal-weight per-bar return of holding every pair (aligned on the shared index)."""
    frame = pd.DataFrame({pair: close.pct_change() for pair, close in closes.items()})
    return frame.mean(axis=1, skipna=True).fillna(0.0)


def sharpe(returns: pd.Series) -> float:
    std = returns.std()
    return float(returns.mean() / std * np.sqrt(BARS_PER_YEAR)) if std > 0 else 0.0


def buy_and_hold(closes: dict[str, pd.Series]) -> dict:
    """Headline stats for holding every pair equally over the given closes."""
    rets = portfolio_returns(closes)
    equity = (1 + rets).cumprod()
    drawdown = (equity / equity.cummax() - 1).min() if len(equity) else 0.0
    return {
        "total_return": round(float(equity.iloc[-1] - 1) if len(equity) else 0.0, 4),
        "sharpe": round(sharpe(rets), 4),
        "max_drawdown": round(float(drawdown), 4),
    }


def alpha_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    """Regress per-bar strategy returns on the benchmark: beta is market exposure,
    alpha (annualised) is the return not explained by riding the market."""
    s, b = strategy_returns.align(benchmark_returns, join="inner")
    var = b.var()
    if len(s) < 2 or var == 0:
        return {"beta": 0.0, "alpha_annual": 0.0}
    beta = float(s.cov(b) / var)
    alpha = float((s.mean() - beta * b.mean()) * BARS_PER_YEAR)
    return {"beta": round(beta, 4), "alpha_annual": round(alpha, 4)}
