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
    ASSUMED_FUNDING_PER_BAR,
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
        """Signals that are long exactly over the up moves make money on every trade."""
        n = 200
        prices = [100.0]
        for i in range(1, n):
            prices.append(prices[-1] * (1.01 if i % 10 < 5 else 0.99))
        close = pd.Series(prices)

        # The signal at bar i sets the position held over bar i+1, so look one bar ahead.
        signals = pd.Series([1 if (i + 1) % 10 < 5 else 0 for i in range(n)])

        m = _compute_metrics(close, signals)
        assert m["trade_count"] == 20
        assert m["win_rate"] == 1.0
        assert m["profit_factor"] == float("inf")
        assert m["total_return"] > 0.5
        assert m["max_drawdown"] > -0.01

    def test_fee_impact(self):
        """On a flat price the only P&L is fees and funding, and both are exact."""
        n = 200
        close = pd.Series(100.0, index=range(n))
        rapid = pd.Series([1 if i % 2 == 0 else 0 for i in range(n)])
        steady = pd.Series(1, index=close.index)

        m_rapid = _compute_metrics(close, rapid)
        m_steady = _compute_metrics(close, steady)

        # rapid: positions over bars 1..199 are 1,0,1,...,1 -> 100 held bars that each pay an
        # entry fee and funding, 99 flat bars that each pay an exit fee
        expected_rapid = (1 - TAKER_FEE - ASSUMED_FUNDING_PER_BAR) ** 100 * (1 - TAKER_FEE) ** 99 - 1
        assert m_rapid["total_return"] == pytest.approx(expected_rapid, abs=1e-4)
        # steady: one entry fee, 199 held bars
        expected_steady = (1 - TAKER_FEE - ASSUMED_FUNDING_PER_BAR) * (1 - ASSUMED_FUNDING_PER_BAR) ** 198 - 1
        assert m_steady["total_return"] == pytest.approx(expected_steady, abs=1e-4)
        assert m_rapid["total_return"] < m_steady["total_return"] - 0.05
        assert m_rapid["trade_count"] == 100
        assert m_steady["trade_count"] == 1

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

    def test_low_win_rate_still_passes(self):
        """Win rate is not gated: a 35% win rate with big winners passes."""
        stats = {
            "total_return": 0.30,
            "max_drawdown": -0.15,
            "profit_factor": 1.5,
            "win_rate": 0.35,
            "trade_count": 50,
            "sharpe": 1.2,
        }
        assert meets_phase1_criteria(stats) is True

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


# --- Engine fixes: fee attribution, size-scaled fees, real funding, look-ahead, signal contract ---

from backtest_api.fast_filter import (  # noqa: E402
    SignalError,
    funding_path,
    load_funding,
    validate_signals,
)


class TestFees:
    def test_exit_fee_belongs_to_the_trade(self):
        """A +0.08% move does not cover two 0.045% fees: the trade is a loss."""
        close = pd.Series([100, 100, 100.08, 100.08, 100.08], dtype=float)
        signals = pd.Series([1, 1, 0, 0, 0])
        m = _compute_metrics(close, signals)

        assert m["total_return"] < 0
        assert m["trade_count"] == 1
        assert m["win_rate"] == 0.0
        assert m["profit_factor"] == 0.0

    def test_flip_costs_two_fees_and_is_two_trades(self):
        close = pd.Series(100.0, index=range(6))
        signals = pd.Series([1, 1, -1, -1, -1, -1])
        zero = pd.Series(0.0, index=close.index)  # no funding, so only fees show
        m = _compute_metrics(close, signals, funding=zero)

        assert m["trade_count"] == 2
        # entry 1x, flip 2x; the last short is still open (no exit fee)
        assert m["total_return"] == pytest.approx((1 - TAKER_FEE) * (1 - 2 * TAKER_FEE) - 1, abs=5e-5)  # metrics are rounded to 4 dp

    def test_flip_trades_each_get_their_fees(self):
        """The long gets its entry + exit fee, the short its entry fee."""
        close = pd.Series([100, 100, 101, 101, 100, 100], dtype=float)
        signals = pd.Series([1, 1, -1, -1, 0, 0])
        zero = pd.Series(0.0, index=close.index)
        m = _compute_metrics(close, signals, funding=zero)
        # long: +1% - 2 fees; short: +~0.99% - 2 fees -> both win
        assert m["trade_count"] == 2
        assert m["win_rate"] == 1.0

    def test_fee_scales_with_size(self):
        close = pd.Series(100.0, index=range(4))
        zero = pd.Series(0.0, index=close.index)
        half = _compute_metrics(close, pd.Series([0.5, 0.5, 0.5, 0.5]), funding=zero)
        full = _compute_metrics(close, pd.Series([1, 1, 1, 1]), funding=zero)
        assert half["total_return"] == pytest.approx(-TAKER_FEE / 2, abs=5e-5)  # metrics are rounded to 4 dp
        assert full["total_return"] == pytest.approx(-TAKER_FEE, abs=5e-5)  # metrics are rounded to 4 dp

    def test_calmar_survives_total_loss(self):
        close = pd.Series([100, 100, 300, 300], dtype=float)
        m = _compute_metrics(close, pd.Series([-1, -1, -1, -1]))  # short through a 3x: < -100%
        assert m["total_return"] <= -1
        assert isinstance(m["calmar"], float)


class TestFunding:
    def test_positive_funding_long_pays_short_receives(self):
        close = pd.Series(100.0, index=range(5))
        funding = pd.Series(0.001, index=close.index)
        long = _compute_metrics(close, pd.Series(1, index=close.index), funding=funding)
        short = _compute_metrics(close, pd.Series(-1, index=close.index), funding=funding)
        # 4 held bars, one entry fee
        assert long["total_return"] == pytest.approx((1 - TAKER_FEE - 0.001) * 0.999 ** 3 - 1, abs=5e-5)  # metrics are rounded to 4 dp
        assert short["total_return"] == pytest.approx((1 - TAKER_FEE + 0.001) * 1.001 ** 3 - 1, abs=5e-5)  # metrics are rounded to 4 dp
        assert short["total_return"] > 0

    def test_nan_funding_bars_use_assumed_rate_on_both_sides(self):
        close = pd.Series(100.0, index=range(3))
        nan = pd.Series(np.nan, index=close.index)
        short = _compute_metrics(close, pd.Series(-1, index=close.index), funding=nan)
        assert short["total_return"] == pytest.approx(
            (1 - TAKER_FEE - ASSUMED_FUNDING_PER_BAR) * (1 - ASSUMED_FUNDING_PER_BAR) - 1, abs=5e-5)  # metrics are rounded to 4 dp

    def test_load_funding_sums_the_hours_paid_during_each_bar(self, tmp_path):
        bars = pd.Series(pd.date_range("2025-01-01", periods=3, freq="4h", tz="UTC"))
        # Payments stamped 01:00..12:00 (each for the hour before). Rate = hour stamp / 1e6.
        hours = pd.date_range("2025-01-01T01:00", periods=12, freq="1h", tz="UTC")
        pd.DataFrame({"timestamp": hours, "rate": [h.hour / 1e6 for h in hours]}).to_feather(
            funding_path(tmp_path, "BTC_USDC-USDC_4h"))

        f = load_funding(tmp_path, "BTC_USDC-USDC_4h", bars)
        # bar 00:00 pays 01..04, bar 04:00 pays 05..08, bar 08:00 pays 09..12
        assert f.tolist() == pytest.approx([10e-6, 26e-6, 42e-6])

    def test_load_funding_marks_bars_outside_the_file_as_missing(self, tmp_path):
        bars = pd.Series(pd.date_range("2025-01-01", periods=3, freq="4h", tz="UTC"))
        hours = pd.date_range("2025-01-01T05:00", periods=4, freq="1h", tz="UTC")
        pd.DataFrame({"timestamp": hours, "rate": [1e-5] * 4}).to_feather(
            funding_path(tmp_path, "BTC_USDC-USDC_4h"))
        f = load_funding(tmp_path, "BTC_USDC-USDC_4h", bars)
        assert np.isnan(f.iloc[0]) and np.isnan(f.iloc[2])
        assert f.iloc[1] == pytest.approx(4e-5)

    def test_missing_file_returns_none(self, tmp_path):
        bars = pd.Series(pd.date_range("2025-01-01", periods=3, freq="4h", tz="UTC"))
        assert load_funding(tmp_path, "BTC_USDC-USDC_4h", bars) is None

    def test_funding_file_name_matches_the_downloader(self, tmp_path):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from download_data import funding_filename, pair_to_filename

        pair = "BTC/USDC:USDC"
        phase1_pair = pair_to_filename(pair, "4h").removesuffix(".feather")
        assert funding_path(tmp_path, phase1_pair) == tmp_path / funding_filename(pair)


def _write_market(data_dir: Path, n: int = 600, seed: int = 1, funding: float | None = None):
    rng = np.random.RandomState(seed)
    prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    ts = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    pd.DataFrame({"timestamp": ts, "open": prices, "high": prices, "low": prices,
                  "close": prices, "volume": 1.0}).to_feather(data_dir / "BTC_USDC-USDC_4h.feather")
    if funding is not None:
        hours = pd.date_range(ts[0] + pd.Timedelta("1h"), ts[-1] + pd.Timedelta("4h"), freq="1h")
        pd.DataFrame({"timestamp": hours, "rate": funding}).to_feather(data_dir / "BTC_USDC-USDC_funding_1h.feather")


def _run(tmp_path, code: str, funding: float | None = None) -> dict:
    (tmp_path / "S.py").write_text(code)
    _write_market(tmp_path, funding=funding)
    return run_fast_filter("S", pairs=["BTC_USDC-USDC_4h"], data_dir=tmp_path, strategies_dir=tmp_path)


EMA_CROSS = '''
def generate_signals(df):
    fast = df["close"].ewm(span=12, adjust=False).mean()
    slow = df["close"].ewm(span=26, adjust=False).mean()
    return (fast > slow).astype(int)
'''

PEEK = '''
def generate_signals(df):
    return (df.close.shift(-1) > df.close).astype(int)
'''


class TestRunFastFilterChecks:
    def test_reports_real_funding(self, tmp_path):
        r = _run(tmp_path, EMA_CROSS, funding=1e-5)
        assert r["per_pair"]["BTC_USDC-USDC_4h"]["funding"] == "real"

    def test_reports_assumed_funding_when_file_missing(self, tmp_path):
        r = _run(tmp_path, EMA_CROSS)
        assert r["per_pair"]["BTC_USDC-USDC_4h"]["funding"] == "assumed"

    def test_real_funding_changes_the_result(self, tmp_path):
        assumed = _run(tmp_path, EMA_CROSS)
        costly = _run(tmp_path, EMA_CROSS, funding=1e-4)
        assert costly["total_return"] < assumed["total_return"]

    def test_causal_strategy_is_not_flagged(self, tmp_path):
        r = _run(tmp_path, EMA_CROSS)
        assert r["lookahead"] is False
        assert "rejected_reason" not in r

    def test_lookahead_strategy_is_rejected(self, tmp_path):
        r = _run(tmp_path, PEEK)
        assert r["lookahead"] is True
        assert r["per_pair"]["BTC_USDC-USDC_4h"]["lookahead"] is True
        assert "Look-ahead" in r["rejected_reason"]
        assert r["sharpe"] > 5  # it would sail through the numeric gates...
        assert meets_phase1_criteria(r) is False  # ...but the gate refuses it

    def test_lookahead_via_full_series_statistic_is_rejected(self, tmp_path):
        """Normalising by the whole-history mean also reads the future."""
        code = '''
def generate_signals(df):
    return (df.close > df.close.mean()).astype(int)
'''
        assert _run(tmp_path, code)["lookahead"] is True

    @pytest.mark.parametrize("body, msg", [
        ("s = (df.close.diff() > 0).astype(float); s.iloc[5] = float('nan'); return s", "NaN"),
        ("return (df.close.diff() > 0).astype(int) * 2", "outside"),
        ("return [1, 0]", "shape"),
    ])
    def test_bad_signals_are_rejected(self, tmp_path, body, msg):
        r = _run(tmp_path, f"def generate_signals(df):\n    {body}\n")
        assert r["invalid_signals"] is True
        assert msg in r["error"]

    def test_gate_rejects_lookahead_stats(self):
        stats = {"total_return": 0.30, "max_drawdown": -0.15, "profit_factor": 1.5,
                 "trade_count": 50, "sharpe": 1.2, "lookahead": True}
        assert meets_phase1_criteria(stats) is False


class TestValidateSignals:
    def test_accepts_bools_lists_and_numpy(self):
        idx = pd.RangeIndex(3)
        assert validate_signals(pd.Series([True, False, True]), idx).tolist() == [1.0, 0.0, 1.0]
        assert validate_signals([1, 0, -1], idx).tolist() == [1.0, 0.0, -1.0]
        assert validate_signals(np.array([0, 0, 1]), idx).tolist() == [0.0, 0.0, 1.0]

    def test_reorders_a_series_to_df_index(self):
        s = pd.Series([1, 0, -1], index=[2, 1, 0])
        assert validate_signals(s, pd.RangeIndex(3)).tolist() == [-1.0, 0.0, 1.0]

    def test_rejects_foreign_index(self):
        with pytest.raises(SignalError, match="index"):
            validate_signals(pd.Series([1, 0, 1], index=[10, 11, 12]), pd.RangeIndex(3))

    def test_rejects_strings(self):
        with pytest.raises(SignalError, match="non-numeric"):
            validate_signals(pd.Series(["buy", "sell", "hold"]), pd.RangeIndex(3))
