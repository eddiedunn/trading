"""Backtest API — FastAPI service for strategy evaluation.

Phase 1: numpy fast filter on the development data (sub-ms per eval)
Phase 2: Freqtrade over three development periods
Phase 3: the final test — one Freqtrade run on the held-back data, once per
         strategy per campaign, only for code that passed Phase 2. A human
         runs it (scripts/submit_strategy.sh); the agent never does.

Every run is logged in the attempt ledger (backtest_api/ledger.py).
"""

import math

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backtest_api import ledger
from backtest_api.fast_filter import STRATEGIES_DIR, meets_phase1_criteria, run_fast_filter
from backtest_api.periods import campaign_id
from backtest_api.walk_forward import walk_forward_test

app = FastAPI(title="Trading Backtest API", version="0.2.0")


def _plain(value):
    """Make metric dicts JSON-safe: numpy scalars to Python, inf/NaN (e.g. no trades) to None."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class BacktestRequest(BaseModel):
    # Becomes a file name and a Freqtrade class name, so keep it an identifier.
    strategy_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    strategy_code: str
    phase: int  # 1 = numpy fast filter, 2 = Freqtrade periods, 3 = final test on held-back data


def _write_strategy(req: BacktestRequest) -> None:
    strat_path = STRATEGIES_DIR / f"{req.strategy_name}.py"
    strat_path.parent.mkdir(parents=True, exist_ok=True)
    strat_path.write_text(req.strategy_code)


def _required_sharpe(attempts: int, stats: dict):
    from backtest_api.fast_filter import required_sharpe

    years = stats.get("years")
    bench = (stats.get("benchmark") or {}).get("sharpe")
    if years is None or bench is None:
        return None
    return required_sharpe(attempts, years, bench)


@app.post("/backtest")
def run_backtest(req: BacktestRequest):
    if req.phase not in (1, 2, 3):
        raise HTTPException(400, "phase must be 1, 2 or 3")
    campaign = campaign_id()
    sha = ledger.code_sha256(req.strategy_code)

    if req.phase == 1:
        _write_strategy(req)
        # Logged before running, so a crashing or invalid strategy still counts as an attempt.
        attempt_id = ledger.record_attempt(campaign, req.strategy_name, sha, 1)
        stats = run_fast_filter(req.strategy_name)
        if stats.get("invalid_signals"):
            raise HTTPException(422, stats["error"])
        if "error" in stats:
            raise HTTPException(500, stats["error"])
        attempts = ledger.attempt_count(campaign)
        passed = bool(meets_phase1_criteria(stats, attempts=attempts))
        ledger.finish_attempt(attempt_id, passed)
        return _plain({
            "phase": 1, "campaign": campaign, "attempts": attempts,
            "required_sharpe": _required_sharpe(attempts, stats), "stats": stats, "passed": passed,
        })

    if req.phase == 2:
        _write_strategy(req)
        attempt_id = ledger.record_attempt(campaign, req.strategy_name, sha, 2)
        result = walk_forward_test(req.strategy_name)
        ledger.finish_attempt(attempt_id, bool(result["passed"]))
        return _plain({"phase": 2, "campaign": campaign, "passed": result["passed"], "windows": result["windows"]})

    # Phase 3: the final test. Checked before the file is written or anything runs.
    if ledger.final_test_done(campaign, sha=sha, name=req.strategy_name):
        raise HTTPException(409, f"{req.strategy_name} or this exact code already had its final test "
                                 f"in campaign {campaign}; each strategy gets one")
    if not ledger.phase2_passed(campaign, sha):
        raise HTTPException(409, f"this exact code has not passed Phase 2 in campaign {campaign}; "
                                 "run Phase 1 and Phase 2 on it first")
    try:
        ledger.claim_final_test(campaign, sha, req.strategy_name)
    except ledger.FinalTestAlreadyRun as e:
        raise HTTPException(409, str(e)) from e

    from backtest_api.final_test import run_final_test

    _write_strategy(req)
    try:
        result = run_final_test(req.strategy_name)
    except Exception:
        ledger.release_final_test(campaign, sha)
        raise
    body = _plain({"phase": 3, "campaign": campaign, "strategy_name": req.strategy_name, **result})
    ledger.record_final_test(campaign, sha, req.strategy_name, bool(result["passed"]), body)
    return body


@app.get("/health")
def health():
    return {"ok": True}
