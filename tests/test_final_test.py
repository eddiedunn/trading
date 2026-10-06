"""Tests for the final test on the held-back data."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from backtest_api import final_test as ft
from backtest_api.walk_forward import CANDLE, MIN_PROFIT_FACTOR, ft_timerange
from tests.test_walk_forward import _ft_time, _write_feathers

HOLDOUT = datetime(2026, 4, 6, tzinfo=timezone.utc)
LAST = datetime(2026, 10, 5, 16, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def box_data(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_DATA_DIR", str(tmp_path / "data"))
    _write_feathers(tmp_path / "data", "2023-12-16 04:00", "2026-10-05 16:00")


def _daily_profit(n_days: int, mean: float, noise: float, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2026-04-06", periods=n_days, freq="D")
    return [[d.strftime("%Y-%m-%d"), float(p)] for d, p in zip(days, mean + rng.normal(0, noise, n_days))]


def _result(*, trades=40, pf=1.6, dd=0.08, profit=0.3, daily=None, end=LAST, start=HOLDOUT):
    return {"strategy": {"S": {
        "total_trades": trades, "profit_factor": pf, "max_drawdown_account": dd,
        "profit_total": profit, "starting_balance": 1000, "sharpe": 99.0,
        "backtest_start": _ft_time(start), "backtest_end": _ft_time(end),
        "daily_profit": daily if daily is not None else _daily_profit(150, 20.0, 5.0),
    }}}


@patch("backtest_api.walk_forward.run_freqtrade_backtest")
def test_runs_one_backtest_from_holdout_to_latest_candle(mock_backtest):
    mock_backtest.return_value = _result()
    out = ft.run_final_test("S")
    mock_backtest.assert_called_once_with("S", ft_timerange(HOLDOUT, LAST))
    assert out["window"]["start"] == HOLDOUT.isoformat()
    assert out["window"]["last_candle"] == LAST.isoformat()
    assert out["window"]["backtest_end"] == _ft_time(LAST)


@patch("backtest_api.walk_forward.run_freqtrade_backtest")
def test_strong_strategy_passes(mock_backtest):
    mock_backtest.return_value = _result()
    out = ft.run_final_test("S")
    assert set(out) == {"passed", "window", "stats", "benchmark", "gate", "final_tests", "required_sharpe"}
    assert set(out["gate"]) == {"total_trades", "profit_factor", "max_drawdown", "profit_total",
                                "sharpe_vs_buy_and_hold"}
    assert out["passed"] is True, out["gate"]
    assert {"total_return", "sharpe", "max_drawdown", "sharpe_daily"} <= set(out["benchmark"])
    # The comparison uses our daily Sharpe, never Freqtrade's reported one.
    assert out["gate"]["sharpe_vs_buy_and_hold"]["value"] == out["stats"]["sharpe_daily"] != 99.0
    assert out["gate"]["sharpe_vs_buy_and_hold"]["threshold"] == out["benchmark"]["sharpe_daily"]


@pytest.mark.parametrize("kwargs, check", [
    ({"trades": 14}, "total_trades"),
    ({"pf": 1.1}, "profit_factor"),
    ({"dd": 0.3}, "max_drawdown"),
    ({"profit": -0.01}, "profit_total"),
    ({"daily": _daily_profit(150, -5.0, 30.0)}, "sharpe_vs_buy_and_hold"),
])
@patch("backtest_api.walk_forward.run_freqtrade_backtest")
def test_each_gate_can_fail(mock_backtest, kwargs, check):
    mock_backtest.return_value = _result(**kwargs)
    out = ft.run_final_test("S")
    assert out["passed"] is False
    assert out["gate"][check]["passed"] is False


@patch("backtest_api.walk_forward.run_freqtrade_backtest")
def test_fifteen_trades_and_no_losing_side_pass(mock_backtest):
    mock_backtest.return_value = _result(trades=15, pf=0)
    out = ft.run_final_test("S")
    assert out["gate"]["total_trades"]["passed"] is True
    assert out["gate"]["profit_factor"] == {"value": None, "threshold": MIN_PROFIT_FACTOR, "passed": True}


@patch("backtest_api.walk_forward.run_freqtrade_backtest")
def test_short_data_fails_with_error(mock_backtest):
    mock_backtest.return_value = _result(end=datetime(2026, 10, 2, 20, tzinfo=timezone.utc))
    out = ft.run_final_test("S")
    assert out["passed"] is False
    assert "ended 2026-10-02 20:00" in out["error"]


@patch("backtest_api.walk_forward.run_freqtrade_backtest")
def test_freqtrade_error_fails(mock_backtest):
    mock_backtest.return_value = {"error": "boom", "returncode": 2}
    out = ft.run_final_test("S")
    assert out["passed"] is False
    assert out["error"] == "boom"
    assert out["gate"] == {} and out["stats"] is None


def test_benchmark_uses_only_holdout_bars():
    closes = ft.holdout_closes(LAST)
    for series in closes.values():
        assert series.index.min() == pd.Timestamp(HOLDOUT)
        assert series.index.max() == pd.Timestamp(LAST)


def test_strategy_daily_returns_fill_quiet_days_and_compound():
    block = {"starting_balance": 100, "daily_profit": [["2026-04-07", 10.0], ["2026-04-09", -11.0]]}
    rets = ft.strategy_daily_returns(block, HOLDOUT, HOLDOUT + timedelta(days=4, hours=20))
    assert list(rets.round(4)) == [0.0, 0.1, 0.0, -0.1, 0.0]


def test_annualised_sharpe_is_daily_sqrt_365():
    r = pd.Series([0.01, 0.02, 0.0, 0.01])
    assert ft.annualised_sharpe(r) == pytest.approx(r.mean() / r.std() * np.sqrt(365))
    assert ft.annualised_sharpe(pd.Series([0.0, 0.0])) == 0.0


def test_sharpe_bar_rises_with_final_tests():
    """The first final test only has to beat buy-and-hold; each later one needs more."""
    from backtest_api.fast_filter import expected_max_sharpe
    bars = [expected_max_sharpe(n, 0.5) for n in (1, 2, 5, 10)]
    assert bars[0] == 0 and bars == sorted(bars) and bars[-1] > bars[1]
