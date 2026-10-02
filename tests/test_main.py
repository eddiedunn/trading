"""Unit tests for the backtest API request handling."""

from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from backtest_api.main import app

client = TestClient(app)


def test_rejects_strategy_name_that_is_not_an_identifier():
    """strategy_name becomes a file path, so path tricks are refused before anything is written."""
    r = client.post("/backtest", json={"strategy_name": "../evil", "strategy_code": "", "phase": 1})
    assert r.status_code == 422


def test_health():
    assert client.get("/health").json() == {"ok": True}


@patch("backtest_api.main.STRATEGIES_DIR")
@patch("backtest_api.main.run_fast_filter")
def test_phase1_response_serializes_numpy_values(mock_filter, mock_dir, tmp_path):
    """The fast filter returns numpy scalars; the API must still answer with JSON."""
    mock_dir.__truediv__.side_effect = lambda name: tmp_path / name
    mock_filter.return_value = {
        "total_return": np.float64(0.25), "sharpe": np.float64(1.1), "max_drawdown": np.float64(-0.1),
        "win_rate": np.float64(0.5), "profit_factor": np.float64(1.5), "trade_count": np.int64(40),
        "calmar": np.float64(np.inf),  # no losing period
    }

    r = client.post("/backtest", json={"strategy_name": "S", "strategy_code": "", "phase": 1})

    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["passed"], bool)
    assert body["stats"]["trade_count"] == 40
    assert body["stats"]["calmar"] is None
