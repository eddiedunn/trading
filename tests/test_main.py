"""Unit tests for the backtest API request handling. The phase runners are mocked;
the ledger is real, in a temp TRADING_RESULTS_DIR."""

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backtest_api import ledger
from backtest_api.main import app

client = TestClient(app)

CAMPAIGN = "2026-04-06"
CODE = "def generate_signals(df):\n    return df['close'] * 0\n"


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr("backtest_api.main.STRATEGIES_DIR", tmp_path / "strategies")
    monkeypatch.setattr("backtest_api.main.campaign_id", lambda: CAMPAIGN)


@pytest.fixture
def phase1(monkeypatch):
    """run_fast_filter, meets_phase1_criteria and required_sharpe, mocked."""
    m = types.SimpleNamespace(
        filter=MagicMock(return_value=_stats()),
        criteria=MagicMock(return_value=True),
        required=MagicMock(return_value=1.25),
    )
    monkeypatch.setattr("backtest_api.main.run_fast_filter", m.filter)
    monkeypatch.setattr("backtest_api.main.meets_phase1_criteria", m.criteria)
    monkeypatch.setattr("backtest_api.fast_filter.required_sharpe", m.required, raising=False)
    return m


@pytest.fixture
def final_test(monkeypatch):
    """backtest_api.final_test.run_final_test, mocked (the module may not exist on this branch)."""
    run = MagicMock(return_value={
        "passed": True, "window": "20260406-20261005",
        "stats": {"sharpe": np.float64(1.4), "total_return": 0.12, "profit_factor": np.float64(np.inf)},
        "benchmark": {"sharpe": 0.6, "total_return": 0.05, "max_drawdown": -0.2},
        "gate": {"sharpe": {"value": 1.4, "threshold": 0.6, "passed": True}},
    })
    monkeypatch.setitem(sys.modules, "backtest_api.final_test", types.SimpleNamespace(run_final_test=run))
    return run


def _stats(**kw):
    return {
        "total_return": np.float64(0.25), "sharpe": np.float64(1.1), "max_drawdown": np.float64(-0.1),
        "win_rate": np.float64(0.5), "profit_factor": np.float64(1.5), "trade_count": np.int64(40),
        "calmar": np.float64(np.inf),  # no losing period
        "years": 2.3, "benchmark": {"total_return": 0.4, "sharpe": 0.7, "max_drawdown": -0.5},
        **kw,
    }


def _post(phase, name="S", code=CODE):
    return client.post("/backtest", json={"strategy_name": name, "strategy_code": code, "phase": phase})


def _pass_phase2(name="S", code=CODE):
    with patch("backtest_api.main.walk_forward_test", return_value={"passed": True, "windows": []}):
        assert _post(2, name, code).json()["passed"] is True


def test_rejects_strategy_name_that_is_not_an_identifier():
    """strategy_name becomes a file path, so path tricks are refused before anything is written."""
    r = client.post("/backtest", json={"strategy_name": "../evil", "strategy_code": "", "phase": 1})
    assert r.status_code == 422


def test_unknown_phase():
    assert _post(4).status_code == 400


def test_health():
    assert client.get("/health").json() == {"ok": True}


class TestPhase1:
    def test_response_serializes_numpy_values(self, phase1):
        """The fast filter returns numpy scalars; the API must still answer with JSON."""
        r = _post(1)
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["passed"], bool)
        assert body["stats"]["trade_count"] == 40
        assert body["stats"]["calmar"] is None

    def test_returns_attempts_and_required_sharpe(self, phase1):
        body = _post(1).json()
        assert body["campaign"] == CAMPAIGN and body["attempts"] == 1 and body["required_sharpe"] == 1.25
        phase1.required.assert_called_once_with(1, 2.3, 0.7)
        assert phase1.criteria.call_args.kwargs == {"attempts": 1}

    def test_attempts_count_distinct_code(self, phase1):
        assert _post(1, code=CODE).json()["attempts"] == 1
        assert _post(1, code=CODE + "   \n\n").json()["attempts"] == 1  # whitespace-only edit
        assert _post(1, name="Other", code=CODE).json()["attempts"] == 1  # same code renamed
        assert _post(1, code=CODE.replace("0", "2")).json()["attempts"] == 2
        assert phase1.criteria.call_args.kwargs == {"attempts": 2}

    def test_failed_run_still_counts(self, phase1):
        phase1.filter.return_value = {"error": "BTC: generate_signals returned 3 NaN values",
                                      "invalid_signals": True, "per_pair": {}}
        r = _post(1)
        assert r.status_code == 422 and "NaN" in r.json()["detail"]
        assert ledger.attempt_count(CAMPAIGN) == 1

    def test_required_sharpe_is_none_without_benchmark(self, phase1):
        phase1.filter.return_value = {k: v for k, v in _stats().items() if k != "benchmark"}
        assert _post(1).json()["required_sharpe"] is None


class TestPhase2:
    @patch("backtest_api.main.walk_forward_test")
    def test_records_result(self, wf):
        wf.return_value = {"passed": True, "windows": [{"label": "period 1", "profit_factor": np.float64(1.4)}]}
        body = _post(2).json()
        assert body["passed"] is True and body["windows"][0]["profit_factor"] == 1.4
        wf.assert_called_once_with("S")
        assert ledger.phase2_passed(CAMPAIGN, ledger.code_sha256(CODE))

    @patch("backtest_api.main.walk_forward_test", return_value={"passed": False, "windows": []})
    def test_failure_is_recorded_as_failure(self, wf):
        _post(2)
        assert not ledger.phase2_passed(CAMPAIGN, ledger.code_sha256(CODE))


class TestFinalTest:
    def test_refused_without_phase2_pass(self, final_test):
        r = _post(3)
        assert r.status_code == 409 and "Phase 2" in r.json()["detail"]
        final_test.assert_not_called()

    def test_refused_when_a_different_code_passed_phase2(self, final_test):
        _pass_phase2(code=CODE)
        r = _post(3, code=CODE.replace("0", "2"))
        assert r.status_code == 409
        final_test.assert_not_called()

    def test_runs_once_and_returns_everything(self, final_test, tmp_path):
        _pass_phase2()
        r = _post(3)
        assert r.status_code == 200
        body = r.json()
        assert body["phase"] == 3 and body["campaign"] == CAMPAIGN and body["passed"] is True
        assert body["window"] == "20260406-20261005"
        assert body["stats"]["sharpe"] == 1.4 and body["stats"]["profit_factor"] is None
        assert body["benchmark"]["sharpe"] == 0.6 and body["gate"]["sharpe"]["passed"] is True
        final_test.assert_called_once_with("S")
        assert (tmp_path / "strategies" / "S.py").read_text() == CODE

    def test_second_final_test_on_same_code_or_name_is_refused(self, final_test):
        _pass_phase2()
        assert _post(3).status_code == 200
        assert _post(3).status_code == 409  # same name, same code
        _pass_phase2(name="Renamed")
        assert _post(3, name="Renamed").status_code == 409  # same code, new name
        other = CODE.replace("0", "2")
        _pass_phase2(code=other)
        r = _post(3, code=other)  # same name, new code
        assert r.status_code == 409 and "already had its final test" in r.json()["detail"]
        assert final_test.call_count == 1

    def test_failing_final_test_still_uses_it_up(self, final_test):
        final_test.return_value = {**final_test.return_value, "passed": False}
        _pass_phase2()
        assert _post(3).json()["passed"] is False
        assert _post(3).status_code == 409

    def test_crash_releases_the_claim(self, final_test):
        _pass_phase2()
        final_test.side_effect = RuntimeError("freqtrade died")
        with pytest.raises(RuntimeError):
            _post(3)
        final_test.side_effect = None
        assert _post(3).status_code == 200
