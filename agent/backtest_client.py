"""Client for the backtest API (`backtest_api/main.py`) plus the feedback
summaries Claude is allowed to see.

Phase 2 feedback deliberately hides the out-of-sample window's numbers and
every window's dates: Claude sees in-sample and validation metrics, and only
pass/fail for out-of-sample, so it cannot tune to the final gate.
"""

import os

import httpx

DEFAULT_API_URL = os.environ.get("BACKTEST_API_URL", "http://127.0.0.1:8070")

PHASE1_TIMEOUT = 300.0
PHASE2_TIMEOUT = 1800.0  # three Freqtrade runs; matches submit_strategy.sh

PHASE1_GATE = "total_return > 20%, max_drawdown > -20%, profit_factor > 1.3, trade_count > 30, sharpe > 0.8"
PHASE2_GATE = "every window: profit_factor >= 1.2 and max_drawdown >= -25%"


class BacktestError(Exception):
    """The API refused or could not run the strategy; message is Claude-safe feedback."""


def run_phase(name: str, code: str, phase: int, api_url: str = DEFAULT_API_URL) -> dict:
    """POST to /backtest. Returns the response body; raises BacktestError on a 4xx/5xx."""
    timeout = PHASE1_TIMEOUT if phase == 1 else PHASE2_TIMEOUT
    r = httpx.post(
        f"{api_url.rstrip('/')}/backtest",
        json={"strategy_name": name, "strategy_code": code, "phase": phase},
        timeout=timeout,
    )
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise BacktestError(f"Phase {phase} could not run (HTTP {r.status_code}): {_trim(str(detail))}")
    return r.json()


def _trim(text: str, limit: int = 1500) -> str:
    return text if len(text) <= limit else text[-limit:]


def _fmt(v) -> str:
    return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def phase1_feedback(result: dict) -> str:
    """Headline Phase 1 metrics as text for the revision prompt."""
    stats = result.get("stats", {})
    lines = [
        f"Phase 1 {'PASSED' if result.get('passed') else 'FAILED'} (gate: {PHASE1_GATE}).",
        "Aggregate across pairs: "
        + ", ".join(f"{k}={_fmt(stats.get(k))}" for k in
                    ("total_return", "max_drawdown", "profit_factor", "win_rate", "trade_count", "sharpe", "calmar")),
    ]
    for pair, s in (stats.get("per_pair") or {}).items():
        lines.append(
            f"  {pair}: return={_fmt(s.get('total_return'))} pf={_fmt(s.get('profit_factor'))} "
            f"dd={_fmt(s.get('max_drawdown'))} trades={_fmt(s.get('trade_count'))}"
        )
    return "\n".join(lines)


def phase2_feedback(result: dict) -> str:
    """Phase 2 summary with the out-of-sample numbers and all dates withheld."""
    lines = [f"Phase 2 {'PASSED' if result.get('passed') else 'FAILED'} (gate: {PHASE2_GATE})."]
    for w in result.get("windows", []):
        label = w.get("label", "?")
        status = "passed" if w.get("passed") else "failed"
        if "error" in w:
            lines.append(f"  {label}: {status} — error: {_trim(str(w['error']), 600)}")
        elif label == "out-of-sample":
            lines.append(f"  {label}: {status}")
        else:
            lines.append(f"  {label}: {status} (pf={_fmt(w.get('profit_factor'))}, dd={_fmt(w.get('max_drawdown'))})")
    return "\n".join(lines)
