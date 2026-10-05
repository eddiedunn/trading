"""The example strategies: they validate clean, run on Phase 1's columns without look-ahead,
and the FundingFade class builds the same funding_rate column as Phase 1."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agent.validate import validate_strategy
from backtest_api.fast_filter import add_context_columns, run_fast_filter

EXAMPLES = Path(__file__).resolve().parent.parent / "strategies" / "examples"
NAMES = ["EmaCross", "FundingFade", "RelativeStrength"]
COINS = ("BTC", "ETH", "SOL")
N = 900


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_market(data_dir: Path):
    """Three coins on one 4h grid, with hourly funding that swings between crowded regimes."""
    ts = pd.date_range("2024-01-01", periods=N, freq="4h", tz="UTC")
    hours = pd.date_range(ts[0] + pd.Timedelta("1h"), ts[-1] + pd.Timedelta("4h"), freq="1h")
    for k, coin in enumerate(COINS):
        rng = np.random.RandomState(40 + k)
        prices = 100 * np.cumprod(1 + rng.normal(0, 0.02, N))
        pd.DataFrame({"timestamp": ts, "open": prices, "high": prices * 1.01, "low": prices * 0.99,
                      "close": prices, "volume": rng.uniform(1, 10, N)}).to_feather(
            data_dir / f"{coin}_USDC-USDC_4h.feather")
        rate = 1.25e-5 + rng.normal(0, 5e-6, len(hours))
        for start in rng.choice(len(hours) - 48, 12, replace=False):  # crowded episodes, either side
            rate[start:start + 48] += rng.choice([-3e-4, 3e-4])
        pd.DataFrame({"timestamp": hours, "rate": rate}).to_feather(data_dir / f"{coin}_USDC-USDC_funding_1h.feather")


@pytest.mark.parametrize("name", NAMES)
def test_example_validates_clean(name):
    assert validate_strategy((EXAMPLES / f"{name}.py").read_text(), name) == []


@pytest.mark.parametrize("name", ["FundingFade", "RelativeStrength"])
def test_generate_signals_on_the_new_columns(tmp_path, name):
    _write_market(tmp_path)
    df = add_context_columns(pd.read_feather(tmp_path / "ETH_USDC-USDC_4h.feather"), "ETH_USDC-USDC_4h", tmp_path)
    sig = _load(name).generate_signals(df.copy())
    assert len(sig) == len(df)
    assert set(sig.unique()) <= {-1, 0, 1}
    assert (sig == 1).any() and (sig == -1).any() and (sig == 0).any()


@pytest.mark.parametrize("name", ["FundingFade", "RelativeStrength"])
def test_example_passes_the_lookahead_check(tmp_path, name):
    _write_market(tmp_path)
    (tmp_path / f"{name}.py").write_text((EXAMPLES / f"{name}.py").read_text())
    r = run_fast_filter(name, pairs=[f"{c}_USDC-USDC_4h" for c in COINS], data_dir=tmp_path,
                        strategies_dir=tmp_path)
    assert "error" not in r
    assert r["lookahead"] is False
    assert r["trade_count"] > 0


def test_funding_fade_with_no_funding_is_flat(tmp_path):
    _write_market(tmp_path)
    for coin in COINS:
        (tmp_path / f"{coin}_USDC-USDC_funding_1h.feather").unlink()
    df = add_context_columns(pd.read_feather(tmp_path / "BTC_USDC-USDC_4h.feather"), "BTC_USDC-USDC_4h", tmp_path)
    assert (_load("FundingFade").generate_signals(df) == 0).all()


def test_funding_fade_class_builds_the_same_funding_as_phase1(tmp_path):
    """The class's funding_rate, built from Freqtrade's hourly funding_rate candles, equals
    Phase 1's column, so both halves compute the same positions."""
    _write_market(tmp_path)
    mod = _load("FundingFade")
    pair = "SOL_USDC-USDC_4h"
    phase1 = add_context_columns(pd.read_feather(tmp_path / f"{pair}.feather"), pair, tmp_path)

    # What Freqtrade hands the strategy: candles with `date`, funding as OHLC rows (download_data.export_freqtrade).
    candles = pd.read_feather(tmp_path / f"{pair}.feather").rename(columns={"timestamp": "date"})
    funding = pd.read_feather(tmp_path / "SOL_USDC-USDC_funding_1h.feather")
    fr = pd.DataFrame({"date": funding["timestamp"], "open": funding["rate"], "high": funding["rate"],
                       "low": funding["rate"], "close": funding["rate"], "volume": 0.0})

    def get_pair_dataframe(p, timeframe, candle_type=""):
        assert (p, timeframe, candle_type) == ("SOL/USDC:USDC", "1h", "funding_rate")
        return fr.copy()

    strat = mod.FundingFade()
    strat.dp = SimpleNamespace(get_pair_dataframe=get_pair_dataframe)
    out = strat.populate_indicators(candles.copy(), {"pair": "SOL/USDC:USDC"})
    # Phase 1 marks the last bar NaN when its funding hours are not all present; compare the rest.
    np.testing.assert_allclose(out["funding_rate"].to_numpy()[:-1], phase1["funding_rate"].to_numpy()[:-1])
    assert (out["pos"].to_numpy() == mod.generate_signals(phase1).to_numpy()).all()
    out = strat.populate_exit_trend(strat.populate_entry_trend(out, {}), {})
    assert ((out["enter_short"] == 1) == (out["pos"] == -1)).all()
    assert ((out["exit_long"] == 1) == (out["pos"] != 1)).all()
