"""Tests for the holdout split and the buy-and-hold benchmark."""

import json

import numpy as np
import pandas as pd
import pytest

from backtest_api import benchmark, periods


@pytest.fixture
def holdout(tmp_path, monkeypatch):
    (tmp_path / "holdout.json").write_text(json.dumps({"holdout_start": "2026-04-06"}))
    monkeypatch.setenv("TRADING_CONFIG_DIR", str(tmp_path))


def test_split_is_exclusive_and_complete(holdout):
    ts = pd.date_range("2026-04-05 16:00", periods=4, freq="4h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "close": [1, 2, 3, 4]})
    dev, hold = periods.development_part(df), periods.holdout_part(df)
    assert list(dev.close) == [1, 2]
    assert list(hold.close) == [3, 4]
    assert periods.campaign_id() == "2026-04-06"


def test_buy_and_hold_equal_weight():
    up = pd.Series([100.0, 110.0, 121.0])
    flat = pd.Series([50.0, 50.0, 50.0])
    stats = benchmark.buy_and_hold({"A": up, "B": flat})
    assert stats["total_return"] == pytest.approx(1.05 * 1.05 - 1, abs=1e-4)
    assert stats["max_drawdown"] == 0


def test_alpha_beta_recovers_leverage():
    rng = np.random.default_rng(0)
    b = pd.Series(rng.normal(0, 0.01, 500))
    ab = benchmark.alpha_beta(2 * b, b)
    assert ab["beta"] == pytest.approx(2.0)
    assert ab["alpha_annual"] == pytest.approx(0.0, abs=1e-9)
