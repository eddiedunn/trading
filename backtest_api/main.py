"""Backtest API — FastAPI service for strategy evaluation.

Phase 1: numpy fast filter (sub-ms per eval)
Phase 2: Freqtrade walk-forward validation
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

from backtest_api.fast_filter import run_fast_filter, meets_phase1_criteria
from backtest_api.walk_forward import walk_forward_test

app = FastAPI(title="Trading Backtest API", version="0.1.0")

_REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGIES_DIR = _REPO_ROOT / "strategies" / "candidates"


class BacktestRequest(BaseModel):
    strategy_name: str
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
        if "error" in stats:
            raise HTTPException(500, stats["error"])
        return {"phase": 1, "stats": stats, "passed": meets_phase1_criteria(stats)}

    if req.phase == 2:
        result = walk_forward_test(req.strategy_name, timerange=req.timerange)
        return {"phase": 2, "passed": result["passed"], "windows": result["windows"]}

    raise HTTPException(400, "phase must be 1 or 2")


@app.get("/health")
def health():
    return {"ok": True}
