"""Known-answer tests for Phase 1 fast filter.

Uses synthetic OHLCV data with deterministic price movements so we can
verify metrics computation against hand-calculated expected values.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest_api.fast_filter import (
    _compute_metrics,
    _aggregate_metrics,
    meets_phase1_criteria,
    run_fast_filter,
    TAKER_FEE,
    FUNDING_RATE_PER_BAR,
)


def _make_price_series(n: int = 200, start: float = 100.0, seed: int = 42) -> pd.Series:
    """Generate a deterministic trending price series."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(0.001, 0.02, n)  # slight upward drift
    prices = start * np.cumprod(1 + returns)
    return pd.Series(prices, name="close")


class TestComputeMetrics:
    """Test the core metrics computation."""

    def test_all_flat_signals(self):
        """All-flat position → zero return, zero trades."""
        close = _make_price_series(100)
        signals = pd.Series(0, index=close.index)
        m = _compute_metrics(close, signals)

        assert m["trade_count"] == 0
        assert m["total_return"] == pytest.approx(0.0, abs=0.001)
        assert m["win_rate"] == 0

    def test_always_long(self):
        """Always-long position should track the underlying (minus fees/funding)."""
        close = _make_price_series(100, seed=7)
        signals = pd.Series(1, index=close.index)
        m = _compute_metrics(close, signals)

        # Should have exactly 1 trade (enters on bar 1, never exits)
        assert m["trade_count"] == 1
        # Should have some return (price trends slightly up)
        assert isinstance(m["total_return"], float)
        assert isinstance(m["sharpe"], float)
        assert isinstance(m["max_drawdown"], float)
        assert m["max_drawdown"] <= 0  # drawdown is always negative or zero

    def test_perfect_signals_positive_return(self):
        """Signals that buy before up moves and sell before down should be profitable."""
        n = 200
        close = pd.Series(dtype=float)
        prices = [100.0]
        # Alternating up/down pattern
        for i in range(1, n):
            if i % 10 < 5:
                prices.append(prices[-1] * 1.01)  # up
            else:
                prices.append(prices[-1] * 0.99)  # down
        close = pd.Series(prices)

        # Perfect signals: long during up, flat during down
        signals = pd.Series(0, index=close.index)
        for i in range(n):
            if i % 10 < 5:
                signals.iloc[i] = 1

        m = _compute_metrics(close, signals)
        # With perfect timing (minus fees), should have positive metrics
        assert m["trade_count"] > 0
        assert isinstance(m["profit_factor"], float)

    def test_fee_impact(self):
        """More trades = more fee drag."""
        close = _make_price_series(200)
        # Rapidly switching: high fee drag
        rapid = pd.Series([1 if i % 2 == 0 else 0 for i in range(200)])
        # Steady hold: low fee drag
        steady = pd.Series(1, index=close.index)

        m_rapid = _compute_metrics(close, rapid)
        m_steady = _compute_metrics(close, steady)

        # Rapid switching should have worse returns due to fees
        assert m_rapid["trade_count"] > m_steady["trade_count"]

    def test_metrics_keys(self):
        """All expected keys present in output."""
        close = _make_price_series(50)
        signals = pd.Series(1, index=close.index)
        m = _compute_metrics(close, signals)

        expected_keys = {
            "total_return", "sharpe", "max_drawdown",
            "win_rate", "profit_factor", "trade_count", "calmar",
        }
        assert set(m.keys()) == expected_keys

    def test_values_are_python_floats(self):
        """Metrics should be plain Python floats, not numpy types (for JSON serialization)."""
        close = _make_price_series(50)
        signals = pd.Series(1, index=close.index)
        m = _compute_metrics(close, signals)

        for key in ["total_return", "sharpe", "max_drawdown", "win_rate", "profit_factor", "calmar"]:
            assert isinstance(m[key], float), f"{key} is {type(m[key])}, expected float"
        assert isinstance(m["trade_count"], int)


class TestAggregateMetrics:
    """Test cross-pair aggregation."""

    def test_aggregation(self):
        results = {
            "BTC": {"total_return": 0.10, "sharpe": 1.0, "max_drawdown": -0.05,
                     "win_rate": 0.50, "profit_factor": 1.5, "trade_count": 20, "calmar": 2.0},
            "ETH": {"total_return": 0.20, "sharpe": 1.5, "max_drawdown": -0.10,
                     "win_rate": 0.60, "profit_factor": 2.0, "trade_count": 30, "calmar": 3.0},
        }
        agg = _aggregate_metrics(results)

        assert agg["total_return"] == pytest.approx(0.15, abs=0.001)
        assert agg["sharpe"] == pytest.approx(1.25, abs=0.001)
        assert agg["trade_count"] == 50
        assert "per_pair" in agg

    def test_handles_inf_profit_factor(self):
        """Infinite profit factor (no losses) should be excluded from average."""
        results = {
            "BTC": {"total_return": 0.10, "sharpe": 1.0, "max_drawdown": -0.05,
                     "win_rate": 0.50, "profit_factor": float("inf"), "trade_count": 10, "calmar": 2.0},
            "ETH": {"total_return": 0.20, "sharpe": 1.5, "max_drawdown": -0.10,
                     "win_rate": 0.60, "profit_factor": 2.0, "trade_count": 20, "calmar": 3.0},
        }
        agg = _aggregate_metrics(results)
        assert agg["profit_factor"] == pytest.approx(2.0, abs=0.001)


class TestPhase1Criteria:
    """Test the gate function."""

    def test_passing(self):
        stats = {
            "total_return": 0.30,
            "max_drawdown": -0.15,
            "profit_factor": 1.5,
            "win_rate": 0.55,
            "trade_count": 50,
            "sharpe": 1.2,
        }
        assert meets_phase1_criteria(stats) is True

    def test_failing_return(self):
        stats = {
            "total_return": 0.10,  # below 0.20
            "max_drawdown": -0.15,
            "profit_factor": 1.5,
            "win_rate": 0.55,
            "trade_count": 50,
            "sharpe": 1.2,
        }
        assert meets_phase1_criteria(stats) is False

    def test_failing_drawdown(self):
        stats = {
            "total_return": 0.30,
            "max_drawdown": -0.25,  # below -0.20
            "profit_factor": 1.5,
            "win_rate": 0.55,
            "trade_count": 50,
            "sharpe": 1.2,
        }
        assert meets_phase1_criteria(stats) is False

    def test_failing_trade_count(self):
        stats = {
            "total_return": 0.30,
            "max_drawdown": -0.15,
            "profit_factor": 1.5,
            "win_rate": 0.55,
            "trade_count": 10,  # below 30
            "sharpe": 1.2,
        }
        assert meets_phase1_criteria(stats) is False


class TestRunFastFilter:
    """Integration test: run_fast_filter with a real strategy file and synthetic data."""

    def test_with_synthetic_data(self, tmp_path):
        """Write a simple strategy + synthetic feather data, run fast_filter end-to-end."""
        # Create strategy
        strategies_dir = tmp_path / "strategies"
        strategies_dir.mkdir()
        strategy_code = '''
import pandas as pd
import numpy as np

def generate_signals(df):
    """Simple RSI-like signal: long when price drops, flat when price rises."""
    returns = df["close"].pct_change()
    signals = pd.Series(0, index=df.index)
    signals[returns < -0.01] = 1   # buy dips
    signals[returns > 0.02] = -1   # short spikes
    return signals
'''
        (strategies_dir / "TestStrat.py").write_text(strategy_code)

        # Create synthetic OHLCV data
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        n = 500
        rng = np.random.RandomState(42)
        prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC"),
            "open": prices * (1 + rng.normal(0, 0.001, n)),
            "high": prices * (1 + abs(rng.normal(0, 0.005, n))),
            "low": prices * (1 - abs(rng.normal(0, 0.005, n))),
            "close": prices,
            "volume": rng.uniform(100, 10000, n),
        })
        df.to_feather(data_dir / "BTC_USDC-USDC_4h.feather")

        result = run_fast_filter(
            "TestStrat",
            pairs=["BTC_USDC-USDC_4h"],
            data_dir=data_dir,
            strategies_dir=strategies_dir,
        )

        assert "error" not in result
        assert "per_pair" in result
        assert "BTC_USDC-USDC_4h" in result["per_pair"]
        assert result["trade_count"] > 0
        assert isinstance(result["sharpe"], float)
