"""Backtest API — FastAPI service for strategy evaluation.

Phase 1: numpy fast filter (sub-ms per eval)
Phase 2: Freqtrade walk-forward validation
"""

import math

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backtest_api.fast_filter import STRATEGIES_DIR, run_fast_filter, meets_phase1_criteria
from backtest_api.walk_forward import walk_forward_test

app = FastAPI(title="Trading Backtest API", version="0.1.0")


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
    phase: int  # 1 = numpy fast filter, 2 = Freqtrade walk-forward
    timerange: str = "20230101-20260101"


@app.post("/backtest")
def run_backtest(req: BacktestRequest):
    # Write strategy code to disk
    strat_path = STRATEGIES_DIR / f"{req.strategy_name}.py"
    strat_path.parent.mkdir(parents=True, exist_ok=True)
    strat_path.write_text(req.strategy_code)

    if req.phase == 1:
        stats = run_fast_filter(req.strategy_name)
        if stats.get("invalid_signals"):
            raise HTTPException(422, stats["error"])
        if "error" in stats:
            raise HTTPException(500, stats["error"])
        return _plain({"phase": 1, "stats": stats, "passed": meets_phase1_criteria(stats)})

    if req.phase == 2:
        result = walk_forward_test(req.strategy_name, timerange=req.timerange)
        return _plain({"phase": 2, "passed": result["passed"], "windows": result["windows"]})

    raise HTTPException(400, "phase must be 1 or 2")


@app.get("/health")
def health():
    return {"ok": True}
