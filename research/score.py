"""Local, free scoring for the research loop (research/program.md).

Runs the Phase 1 fast filter on this machine only, never on tela, so nothing
here is logged as a campaign attempt. One strategy file is scored from several
views and the headline score is the WORST of them: a fit to noise in one
period or one coin cannot carry the whole number.

Views:
- the full development history as one account (what Phase 1 on tela computes)
- each of the three Phase 2 periods (same dates as walk_forward.default_windows,
  plus its 20-day warm-up in front), run through the same fast filter
- each pair on its own over the full history

Usage: uv run python -m research.score [path/to/Candidate.py]
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

from agent.validate import validate_strategy
from backtest_api import walk_forward
from backtest_api.fast_filter import DATA_DIR, DEFAULT_PAIRS, funding_path, run_fast_filter

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATE = _REPO_ROOT / "research" / "Candidate.py"

# Phase 2 gates, repeated here so each period's line shows them.
PERIOD_GATE = {
    "profit_factor": walk_forward.MIN_PROFIT_FACTOR,
    "max_drawdown": walk_forward.MAX_DRAWDOWN,
    "trade_count": walk_forward.MIN_TRADES,
}


def _slice_data(src: Path, dst: Path, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Copy the pairs' bars in [start, end) and their funding files into dst."""
    for pair in DEFAULT_PAIRS:
        df = pd.read_feather(src / f"{pair}.feather")
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df[(ts >= start) & (ts < end)].reset_index(drop=True).to_feather(dst / f"{pair}.feather")
        fp = funding_path(src, pair)
        if fp.exists():
            shutil.copy(fp, funding_path(dst, pair))


def period_views(strategy_name: str, strategies_dir: Path, data_dir: Path) -> list[dict]:
    """Fast-filter stats for each Phase 2 period (with warm-up bars in front)."""
    first = walk_forward.common_candle_range()[0]
    out = []
    for start, end, label in walk_forward.default_windows(first):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _slice_data(data_dir, tmp, pd.Timestamp(start - walk_forward.WARMUP), pd.Timestamp(end))
            stats = run_fast_filter(strategy_name, data_dir=tmp, strategies_dir=strategies_dir)
        stats["label"] = label
        stats["start"], stats["end"] = start.date().isoformat(), end.date().isoformat()
        out.append(stats)
    return out


def score_file(path: Path, data_dir: Path | None = None) -> dict:
    data_dir = data_dir or DATA_DIR
    name = path.stem
    problems = validate_strategy(path.read_text(), name)
    if problems:
        return {"error": "validation: " + "; ".join(problems)}

    full = run_fast_filter(name, data_dir=data_dir, strategies_dir=path.parent)
    if "error" in full:
        return {"error": full["error"]}
    if full.get("lookahead"):
        return {"error": full["rejected_reason"]}

    periods = period_views(name, path.parent, data_dir)
    per_pair = full["per_pair"]

    views = {p["label"]: p["sharpe"] for p in periods}
    views.update({pair.split("_")[0]: m["sharpe"] for pair, m in per_pair.items()})
    worst = min(views, key=views.get)
    return {
        "score": views[worst],
        "worst_view": worst,
        "views": views,
        "full": full,
        "periods": periods,
    }


def _fails(gate: dict) -> list[str]:
    return [k for k, v in gate.items() if isinstance(v, dict) and not v.get("passed", True)]


def _period_fails(p: dict) -> list[str]:
    bad = []
    if p["profit_factor"] < PERIOD_GATE["profit_factor"]:
        bad.append("profit_factor")
    if p["max_drawdown"] < PERIOD_GATE["max_drawdown"]:
        bad.append("max_drawdown")
    if p["trade_count"] < PERIOD_GATE["trade_count"]:
        bad.append("trade_count")
    if p["total_return"] <= 0:
        bad.append("return")
    return bad


def summary(result: dict) -> str:
    lines = ["---"]
    if "error" in result:
        lines.append(f"score:            0.0000")
        lines.append(f"error:            {result['error']}")
        return "\n".join(lines)
    full = result["full"]
    lines.append(f"score:            {result['score']:.4f}   (worst-view Sharpe; view = {result['worst_view']})")
    gate_fails = _fails(full["gate"])
    lines.append(f"phase1_gate:      {'PASS' if not gate_fails else 'FAIL ' + ','.join(gate_fails)}")
    lines.append(
        f"full:             sharpe {full['sharpe']:.2f}  cagr {full['cagr']:+.1%}  dd {full['max_drawdown']:.1%}"
        f"  pf {full['profit_factor']:.2f}  trades {full['trade_count']}  beta {full['alpha_beta'].get('beta', float('nan')):.2f}"
    )
    lines.append(f"benchmark_sharpe: {full['benchmark']['sharpe']:.2f}   (buy-and-hold; Phase 1 bar is max(0.8, this))")
    for p in result["periods"]:
        fails = _period_fails(p)
        lines.append(
            f"{p['label'].replace(' ', '_')}:         sharpe {p['sharpe']:.2f}  ret {p['total_return']:+.1%}"
            f"  dd {p['max_drawdown']:.1%}  pf {p['profit_factor']:.2f}  trades {p['trade_count']}"
            f"  [{p['start']}..{p['end']}]  {'ok' if not fails else 'FAIL ' + ','.join(fails)}"
        )
    for pair, m in full["per_pair"].items():
        coin = pair.split("_")[0]
        lines.append(
            f"pair_{coin}:         sharpe {m['sharpe']:.2f}  ret {m['total_return']:+.1%}  dd {m['max_drawdown']:.1%}"
            f"  pf {m['profit_factor']:.2f}  trades {m['trade_count']}  funding {m['funding']}"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_CANDIDATE
    result = score_file(path.resolve())
    print(summary(result))
    (path.parent / "last_score.json").write_text(json.dumps(result, indent=2, default=str))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
