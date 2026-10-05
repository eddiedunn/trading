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
    _portfolio_stats,
    expected_max_sharpe,
    floor_failures,
    meets_phase1_criteria,
    phase1_gate,
    portfolio_net_returns,
    required_sharpe,
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


def _pair(pf=1.5, dd=-0.10):
    return {"total_return": 0.1, "sharpe": 1.0, "max_drawdown": dd, "win_rate": 0.5,
            "profit_factor": pf, "trade_count": 20, "calmar": 1.0}


class TestPortfolio:
    """One combined account, equal weight, rebalanced every bar."""

    def test_net_return_is_mean_of_pairs_aligned_on_timestamp(self):
        ts = pd.date_range("2025-01-01", periods=3, freq="4h", tz="UTC")
        a = pd.Series([0.0, 0.02, -0.01], index=ts)
        b = pd.Series([0.01, 0.04], index=ts[1:])  # starts one bar later
        p = portfolio_net_returns({"A": a, "B": b})
        assert p.tolist() == pytest.approx([0.0, 0.015, 0.015])

    def test_headline_metrics_come_from_the_portfolio_not_averages(self):
        ts = pd.date_range("2025-01-01", periods=4, freq="4h", tz="UTC")
        # Pair A: +10% then -10%; pair B: -10% then +10%. Averaging per-pair stats would show
        # a -1% return and -10% drawdown; the rebalanced account is flat throughout.
        a = pd.Series([0.0, 0.10, -0.10, 0.0], index=ts)
        b = pd.Series([0.0, -0.10, 0.10, 0.0], index=ts)
        port = portfolio_net_returns({"A": a, "B": b})
        stats = _portfolio_stats(port, pd.Series([0.05, -0.02, 0.03]))
        assert stats["total_return"] == pytest.approx(0.0)
        assert stats["max_drawdown"] == pytest.approx(0.0)
        # trade stats are pooled across pairs
        assert stats["trade_count"] == 3
        assert stats["profit_factor"] == pytest.approx(4.0)
        assert stats["win_rate"] == pytest.approx(0.6667, abs=1e-4)

    def test_cagr_uses_timestamps(self):
        ts = pd.date_range("2024-01-01", "2026-01-01", periods=101, tz="UTC")
        port = pd.Series(0.0, index=ts)
        port.iloc[1] = 0.21  # +21% over ~2 years
        stats = _portfolio_stats(port, pd.Series(dtype=float))
        assert stats["years"] == pytest.approx(2.0, abs=0.01)
        assert stats["cagr"] == pytest.approx(0.10, abs=0.001)
        assert stats["calmar"] == 0.0  # no drawdown

    def test_pooled_profit_factor_inf_without_losses(self):
        port = pd.Series([0.0, 0.01], index=pd.date_range("2025-01-01", periods=2, freq="4h", tz="UTC"))
        assert _portfolio_stats(port, pd.Series([0.01, 0.02]))["profit_factor"] == float("inf")


class TestFloor:
    def test_all_pairs_pass(self):
        assert floor_failures({"A": _pair(), "B": _pair(pf=1.0, dd=-0.34)}) == []

    def test_reports_the_failing_pair(self):
        f = floor_failures({"A": _pair(pf=3.0), "B": _pair(pf=0.9), "C": _pair(dd=-0.35)})
        assert [(x["pair"], x["check"]) for x in f] == [("B", "profit_factor"), ("C", "max_drawdown")]
        assert f[0]["value"] == 0.9 and f[0]["threshold"] == 1.0


class TestRequiredSharpe:
    def test_one_attempt_has_no_noise_bar(self):
        assert expected_max_sharpe(1, 2.3) == 0.0
        assert required_sharpe(1, 2.3, 0.0) == pytest.approx(0.8)

    def test_rises_with_attempts(self):
        vals = [required_sharpe(n, 2.3, 0.0) for n in (1, 2, 5, 10, 100, 1000)]
        assert all(b > a for a, b in zip(vals, vals[1:]))

    def test_falls_with_more_years(self):
        assert required_sharpe(10, 4.0, 0.0) < required_sharpe(10, 1.0, 0.0)

    def test_floored_at_0_8(self):
        assert required_sharpe(1, 2.3, -1.5) == pytest.approx(0.8)
        assert required_sharpe(10, 2.3, 0.2) == pytest.approx(0.8 + expected_max_sharpe(10, 2.3))

    def test_respects_the_benchmark(self):
        assert required_sharpe(1, 2.3, 1.4) == pytest.approx(1.4)
        assert required_sharpe(10, 2.3, 1.4) == pytest.approx(1.4 + expected_max_sharpe(10, 2.3))

    def test_matches_expected_max_of_normals(self):
        # E[max of 10 standard normals] is 1.5388; the Bailey & Lopez de Prado form is close.
        assert expected_max_sharpe(10, 1.0) == pytest.approx(1.5388, abs=0.05)


def _passing_stats(**overrides):
    stats = {
        "cagr": 0.25, "total_return": 0.60, "max_drawdown": -0.15, "profit_factor": 1.5,
        "win_rate": 0.55, "trade_count": 50, "sharpe": 1.2, "years": 2.3,
        "benchmark": {"total_return": 0.5, "sharpe": 0.6, "max_drawdown": -0.5},
        "floor_failures": [], "lookahead": False,
    }
    stats.update(overrides)
    return stats


class TestPhase1Criteria:
    """Test the gate function."""

    def test_passing(self):
        assert meets_phase1_criteria(_passing_stats()) is True

    @pytest.mark.parametrize("key, value", [
        ("cagr", 0.10),             # not above 10%
        ("max_drawdown", -0.25),    # below -20%
        ("profit_factor", 1.3),     # not above 1.3
        ("trade_count", 30),        # not above 30
        ("sharpe", 0.8),            # not above 0.8
        ("lookahead", True),
        ("floor_failures", [{"pair": "SOL", "check": "profit_factor", "value": 0.9, "threshold": 1.0}]),
    ])
    def test_each_check_can_fail_it(self, key, value):
        stats = _passing_stats(**{key: value})
        assert meets_phase1_criteria(stats) is False
        failed = [k for k, c in stats["gate"].items() if not c["passed"]]
        assert len(failed) == 1

    def test_low_win_rate_still_passes(self):
        """Win rate is not gated: a 35% win rate with big winners passes."""
        assert meets_phase1_criteria(_passing_stats(win_rate=0.35)) is True

    def test_more_attempts_raise_the_sharpe_bar(self):
        stats = _passing_stats(sharpe=1.2)
        assert meets_phase1_criteria(stats, attempts=1) is True
        assert meets_phase1_criteria(stats, attempts=10) is False
        sharpe_check = stats["gate"]["sharpe"]
        assert sharpe_check["required_sharpe"] == pytest.approx(required_sharpe(10, 2.3, 0.6), abs=1e-4)
        assert sharpe_check["passed"] is False and sharpe_check["attempts"] == 10

    def test_strategy_must_beat_buy_and_hold_sharpe(self):
        stats = _passing_stats(sharpe=1.2, benchmark={"sharpe": 1.3})
        assert meets_phase1_criteria(stats) is False
        assert stats["gate"]["sharpe"]["threshold"] == pytest.approx(1.3)

    def test_gate_breakdown_has_value_threshold_passed(self):
        gate = phase1_gate(_passing_stats())
        assert set(gate) == {"cagr", "max_drawdown", "profit_factor", "trade_count",
                             "sharpe", "floor", "lookahead"}
        for check in gate.values():
            assert {"value", "threshold", "passed"} <= set(check)

    def test_missing_floor_result_fails_closed(self):
        stats = _passing_stats()
        del stats["floor_failures"]
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
        assert meets_phase1_criteria(_passing_stats(lookahead=True)) is False


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


# --- Development data only, combined account, benchmark, gate breakdown ---

def _write_pair(data_dir: Path, pair: str, start: str, n: int, seed: int):
    rng = np.random.RandomState(seed)
    prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    ts = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    pd.DataFrame({"timestamp": ts, "open": prices, "high": prices, "low": prices,
                  "close": prices, "volume": 1.0}).to_feather(data_dir / f"{pair}.feather")


@pytest.fixture
def holdout_2025_03_01(tmp_path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    (config / "holdout.json").write_text('{"holdout_start": "2025-03-01"}')
    monkeypatch.setenv("TRADING_CONFIG_DIR", str(config))
    return pd.Timestamp("2025-03-01", tz="UTC")


class TestDevelopmentDataOnly:
    def test_strategy_that_trades_only_after_holdout_start_has_zero_trades(self, tmp_path, holdout_2025_03_01):
        # 600 bars from 2025-01-01 run to mid-April: the last ~280 bars are holdout.
        _write_pair(tmp_path, "BTC_USDC-USDC_4h", "2025-01-01", 600, seed=3)
        (tmp_path / "Late.py").write_text(f'''
import pandas as pd
def generate_signals(df):
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return (ts >= pd.Timestamp("{holdout_2025_03_01.isoformat()}")).astype(int)
''')
        r = run_fast_filter("Late", pairs=["BTC_USDC-USDC_4h"], data_dir=tmp_path, strategies_dir=tmp_path)
        assert r["trade_count"] == 0
        assert r["total_return"] == 0.0
        assert r["lookahead"] is False
        assert pd.Timestamp(r["development_end"]) < holdout_2025_03_01

    def test_generate_signals_never_sees_a_holdout_bar(self, tmp_path, holdout_2025_03_01):
        _write_pair(tmp_path, "BTC_USDC-USDC_4h", "2025-01-01", 600, seed=3)
        (tmp_path / "Guard.py").write_text(f'''
import pandas as pd
def generate_signals(df):
    assert pd.to_datetime(df["timestamp"], utc=True).max() < pd.Timestamp("{holdout_2025_03_01.isoformat()}")
    return (df.close.diff() > 0).astype(int)
''')
        r = run_fast_filter("Guard", pairs=["BTC_USDC-USDC_4h"], data_dir=tmp_path, strategies_dir=tmp_path)
        assert "error" not in r
        assert r["trade_count"] > 0

    def test_pair_with_only_holdout_data_is_skipped(self, tmp_path, holdout_2025_03_01):
        _write_pair(tmp_path, "BTC_USDC-USDC_4h", "2025-04-01", 100, seed=3)
        (tmp_path / "S.py").write_text(EMA_CROSS)
        r = run_fast_filter("S", pairs=["BTC_USDC-USDC_4h"], data_dir=tmp_path, strategies_dir=tmp_path)
        assert r["error"] == "No data files found"


class TestCombinedAccount:
    PAIRS = ["BTC_USDC-USDC_4h", "ETH_USDC-USDC_4h", "SOL_USDC-USDC_4h"]

    def _run(self, tmp_path, code=EMA_CROSS, attempts=1):
        for seed, pair in enumerate(self.PAIRS, start=11):
            _write_pair(tmp_path, pair, "2023-01-01", 800, seed=seed)
        (tmp_path / "S.py").write_text(code)
        return run_fast_filter("S", pairs=self.PAIRS, data_dir=tmp_path, strategies_dir=tmp_path,
                               attempts=attempts)

    def test_headline_is_the_equal_weight_account(self, tmp_path):
        from backtest_api.fast_filter import _simulate
        r = self._run(tmp_path)
        nets = {}
        for pair in self.PAIRS:
            df = pd.read_feather(tmp_path / f"{pair}.feather")
            sig = (df.close.ewm(span=12, adjust=False).mean() > df.close.ewm(span=26, adjust=False).mean()).astype(int)
            nets[pair] = pd.Series(_simulate(df.close, sig)[0].to_numpy(), index=df.timestamp)
        expected = float((1 + pd.DataFrame(nets).mean(axis=1)).prod() - 1)
        assert r["total_return"] == pytest.approx(expected, abs=1e-4)
        assert r["trade_count"] == sum(m["trade_count"] for m in r["per_pair"].values())
        assert set(r["per_pair"]) == set(self.PAIRS)
        assert {"cagr", "sharpe", "max_drawdown", "calmar", "profit_factor", "win_rate", "years"} <= set(r)

    def test_reports_benchmark_alpha_beta_floor_and_gate(self, tmp_path):
        r = self._run(tmp_path)
        assert set(r["benchmark"]) == {"total_return", "sharpe", "max_drawdown"}
        assert set(r["alpha_beta"]) == {"beta", "alpha_annual"}
        assert 0 < r["alpha_beta"]["beta"] < 1  # long-or-flat: partial market exposure
        assert r["floor_failures"] == floor_failures(r["per_pair"])
        assert r["gate"]["sharpe"]["required_sharpe"] == pytest.approx(
            required_sharpe(1, r["years"], r["benchmark"]["sharpe"]), abs=1e-4)
        assert meets_phase1_criteria(r) is all(c["passed"] for c in r["gate"].values())

    def test_attempts_flow_into_the_gate(self, tmp_path):
        one = self._run(tmp_path, attempts=1)["gate"]["sharpe"]["required_sharpe"]
        many = self._run(tmp_path, attempts=50)["gate"]["sharpe"]["required_sharpe"]
        assert many > one

    def test_floor_failure_names_the_pair(self, tmp_path):
        # Long only on SOL every other bar: fees make SOL lose; BTC/ETH flat.
        code = '''
import numpy as np
def generate_signals(df):
    if df.close.iloc[0] != SOL_FIRST:
        return df.close * 0
    return (np.arange(len(df)) % 2).astype(int)
'''
        sol_first = None
        for seed, pair in enumerate(self.PAIRS, start=11):
            _write_pair(tmp_path, pair, "2023-01-01", 800, seed=seed)
        sol_first = float(pd.read_feather(tmp_path / "SOL_USDC-USDC_4h.feather").close.iloc[0])
        (tmp_path / "S.py").write_text(code.replace("SOL_FIRST", repr(sol_first)))
        r = run_fast_filter("S", pairs=self.PAIRS, data_dir=tmp_path, strategies_dir=tmp_path)
        assert any(f["pair"] == "SOL_USDC-USDC_4h" and f["check"] == "profit_factor"
                   for f in r["floor_failures"])
        assert r["gate"]["floor"]["passed"] is False
        assert meets_phase1_criteria(r) is False


# --- Extra columns: funding_rate, close_<COIN>, funding_<COIN> ---

from backtest_api.fast_filter import add_context_columns  # noqa: E402

COINS = ("BTC", "ETH", "SOL")


def _write_three(data_dir: Path, n: int = 300, with_funding: bool = True):
    """Three pairs on the same 4h grid; hourly funding whose value encodes its own stamp."""
    ts = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    for k, coin in enumerate(COINS):
        rng = np.random.RandomState(20 + k)
        prices = 100 * (k + 1) * np.cumprod(1 + rng.normal(0, 0.02, n))
        pd.DataFrame({"timestamp": ts, "open": prices, "high": prices * 1.01, "low": prices * 0.99,
                      "close": prices, "volume": 1.0}).to_feather(data_dir / f"{coin}_USDC-USDC_4h.feather")
        if with_funding:
            hours = pd.date_range(ts[0] + pd.Timedelta("1h"), ts[-1] + pd.Timedelta("4h"), freq="1h")
            # rate = (hours since start) * 1e-6 + coin offset, so every sum is checkable by hand
            rate = (np.arange(len(hours)) + 1) * 1e-6 + k * 1e-3
            pd.DataFrame({"timestamp": hours, "rate": rate}).to_feather(
                data_dir / f"{coin}_USDC-USDC_funding_1h.feather")
    return ts


class TestContextColumns:
    def _df(self, tmp_path, pair="ETH_USDC-USDC_4h", **kw):
        _write_three(tmp_path, **kw)
        df = pd.read_feather(tmp_path / f"{pair}.feather")
        return add_context_columns(df, pair, tmp_path)

    def test_columns_present(self, tmp_path):
        df = self._df(tmp_path)
        for col in ("funding_rate", "close_BTC", "close_ETH", "close_SOL",
                    "funding_BTC", "funding_ETH", "funding_SOL"):
            assert col in df.columns

    def test_funding_rate_is_the_four_payments_paid_by_the_bars_close(self, tmp_path):
        """Bar opening at T gets the payments stamped T+1h..T+4h, never T+5h or later."""
        df = self._df(tmp_path)
        start = pd.Timestamp("2024-01-01", tz="UTC")
        for i in (0, 1, 50, 299):
            t = pd.to_datetime(df["timestamp"].iloc[i], utc=True)
            stamps = [t + pd.Timedelta(hours=h) for h in range(1, 5)]
            # ETH is coin index 1: offset 1e-3 per hour
            expected = sum(((s - start) / pd.Timedelta("1h")) * 1e-6 + 1e-3 for s in stamps)
            assert df["funding_rate"].iloc[i] == pytest.approx(expected, rel=1e-9)
            assert max(stamps) == t + pd.Timedelta("4h")  # the bar's close: nothing later

    def test_other_coins_aligned_on_timestamp(self, tmp_path):
        df = self._df(tmp_path)
        btc = pd.read_feather(tmp_path / "BTC_USDC-USDC_4h.feather")
        assert df["close_BTC"].tolist() == pytest.approx(btc["close"].tolist())
        assert df["close_ETH"].tolist() == pytest.approx(df["close"].tolist())
        assert df["funding_ETH"].tolist() == pytest.approx(df["funding_rate"].tolist())
        assert (df["funding_SOL"] - df["funding_BTC"]).round(9).eq(4 * 2e-3).all()

    def test_misaligned_other_coin_is_matched_by_time_not_row(self, tmp_path):
        _write_three(tmp_path)
        btc = pd.read_feather(tmp_path / "BTC_USDC-USDC_4h.feather").iloc[10:].reset_index(drop=True)
        btc.to_feather(tmp_path / "BTC_USDC-USDC_4h.feather")
        df = add_context_columns(pd.read_feather(tmp_path / "ETH_USDC-USDC_4h.feather"),
                                 "ETH_USDC-USDC_4h", tmp_path)
        assert df["close_BTC"].iloc[:10].isna().all()
        assert df["close_BTC"].iloc[10] == pytest.approx(btc["close"].iloc[0])

    def test_missing_funding_is_nan(self, tmp_path):
        df = self._df(tmp_path, with_funding=False)
        assert df["funding_rate"].isna().all()
        assert df["funding_BTC"].isna().all()

    def test_missing_other_pair_is_nan(self, tmp_path):
        _write_three(tmp_path)
        (tmp_path / "SOL_USDC-USDC_4h.feather").unlink()
        df = add_context_columns(pd.read_feather(tmp_path / "BTC_USDC-USDC_4h.feather"),
                                 "BTC_USDC-USDC_4h", tmp_path)
        assert df["close_SOL"].isna().all()

    def test_prefix_of_columns_equals_columns_of_prefix(self, tmp_path):
        """No look-ahead in the alignment: row t never depends on bars after t."""
        _write_three(tmp_path)
        full = pd.read_feather(tmp_path / "SOL_USDC-USDC_4h.feather")
        whole = add_context_columns(full, "SOL_USDC-USDC_4h", tmp_path)
        cut = 120
        # Rewrite every file truncated at the cut (funding through that bar's close only).
        last_close = pd.to_datetime(full["timestamp"].iloc[cut - 1], utc=True) + pd.Timedelta("4h")
        for coin in COINS:
            p = tmp_path / f"{coin}_USDC-USDC_4h.feather"
            pd.read_feather(p).iloc[:cut].to_feather(p)
            f = tmp_path / f"{coin}_USDC-USDC_funding_1h.feather"
            fr = pd.read_feather(f)
            fr[pd.to_datetime(fr["timestamp"], utc=True) <= last_close].reset_index(drop=True).to_feather(f)
        part = add_context_columns(full.iloc[:cut], "SOL_USDC-USDC_4h", tmp_path)
        cols = ["funding_rate", "close_BTC", "close_ETH", "funding_BTC", "funding_ETH"]
        pd.testing.assert_frame_equal(whole[cols].iloc[:cut], part[cols])


class TestStrategiesSeeContextColumns:
    PAIRS = ["BTC_USDC-USDC_4h", "ETH_USDC-USDC_4h", "SOL_USDC-USDC_4h"]

    def _run(self, tmp_path, code):
        _write_three(tmp_path)
        (tmp_path / "S.py").write_text(code)
        return run_fast_filter("S", pairs=self.PAIRS, data_dir=tmp_path, strategies_dir=tmp_path)

    def test_causal_use_of_funding_and_other_coins_passes_the_prefix_check(self, tmp_path):
        r = self._run(tmp_path, '''
def generate_signals(df):
    hot = df["funding_rate"].rolling(6).sum() > df["funding_rate"].rolling(30).sum() / 5
    btc_up = df["close_BTC"].pct_change(6, fill_method=None) > 0
    return (hot.astype(int) - btc_up.astype(int))
''')
        assert "error" not in r
        assert r["lookahead"] is False
        assert all(m["funding"] == "real" for m in r["per_pair"].values())

    def test_peeking_at_next_bars_funding_is_rejected(self, tmp_path):
        r = self._run(tmp_path, '''
def generate_signals(df):
    return (df["funding_rate"].shift(-1) > df["funding_rate"]).astype(int)
''')
        assert r["lookahead"] is True

    def test_signal_reads_only_one_pair_but_sees_all_three(self, tmp_path):
        r = self._run(tmp_path, '''
def generate_signals(df):
    for c in ("close_BTC", "close_ETH", "close_SOL", "funding_BTC", "funding_ETH", "funding_SOL"):
        assert c in df.columns
    return (df["close_ETH"] > df["close_ETH"].shift(1)).astype(int)
''')
        assert "error" not in r
        assert r["lookahead"] is False
