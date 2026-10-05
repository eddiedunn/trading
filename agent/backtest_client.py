"""Client for the backtest API (`backtest_api/main.py`) plus the feedback
summaries Claude sees.

The agent can only run Phase 1 and Phase 2, which use development data, so
their results are shown in full. The final test (phase 3, the held-back data)
is run by a human through scripts/submit_strategy.sh; run_phase refuses it.
"""

import os

import httpx

DEFAULT_API_URL = os.environ.get("BACKTEST_API_URL", "http://127.0.0.1:8070")

PHASE1_TIMEOUT = 300.0
PHASE2_TIMEOUT = 1800.0  # three Freqtrade runs; matches submit_strategy.sh

AGENT_PHASES = (1, 2)  # never 3: the final test belongs to a human

PHASE1_GATE = (
    "floors on total_return, max_drawdown, profit_factor and trade_count; a Sharpe above a bar that rises "
    "with every attempt in the campaign; and a Sharpe above buy-and-hold's over the same bars. Each result "
    "lists every check with its value and threshold"
)
PHASE2_GATE = "every period: profit_factor >= 1.2 and max_drawdown >= -25%"


class BacktestError(Exception):
    """The API refused or could not run the strategy; message is Claude-safe feedback."""


def run_phase(name: str, code: str, phase: int, api_url: str = DEFAULT_API_URL) -> dict:
    """POST to /backtest. Returns the response body; raises BacktestError on a 4xx/5xx."""
    if phase not in AGENT_PHASES:
        raise ValueError(f"the agent may only run phases {AGENT_PHASES}; the final test is run by a human")
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


def _list(value) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    return str(value)


def phase1_feedback(result: dict) -> str:
    """Phase 1 as text for the revision prompt: the gate check by check, the
    buy-and-hold comparison, the campaign's attempt count and Sharpe bar, then the metrics."""
    stats = result.get("stats", {})
    lines = [f"Phase 1 {'PASSED' if result.get('passed') else 'FAILED'} (gate: {PHASE1_GATE})."]
    if result.get("attempts") is not None:
        lines.append(
            f"Campaign attempts so far: {result['attempts']} effective ({result.get('ideas')} ideas, "
            f"{result.get('versions')} versions; a revision counts as a quarter of a new idea); "
            f"the Sharpe bar is now {_fmt(result.get('required_sharpe'))} and rises with every new attempt."
        )
    gate = stats.get("gate") or {}
    if gate:
        lines.append("Gate checks:")
        for check, g in gate.items():
            g = g or {}
            lines.append(f"  {check}: {_fmt(g.get('value'))} vs threshold {_fmt(g.get('threshold'))} — "
                         f"{'passed' if g.get('passed') else 'FAILED'}")
    if stats.get("floor_failures"):
        lines.append(f"Floor failures: {_list(stats['floor_failures'])}")
    bench = stats.get("benchmark") or {}
    if bench:
        ab = stats.get("alpha_beta") or {}
        lines.append(
            f"Buy-and-hold of every pair over the same bars: sharpe={_fmt(bench.get('sharpe'))}, "
            f"total_return={_fmt(bench.get('total_return'))}, max_drawdown={_fmt(bench.get('max_drawdown'))}. "
            f"Strategy sharpe={_fmt(stats.get('sharpe'))} vs buy-and-hold {_fmt(bench.get('sharpe'))}; "
            f"beta={_fmt(ab.get('beta'))}, alpha_annual={_fmt(ab.get('alpha_annual'))}."
        )
    lines.append(
        "Aggregate across pairs: "
        + ", ".join(f"{k}={_fmt(stats.get(k))}" for k in
                    ("total_return", "cagr", "max_drawdown", "profit_factor", "win_rate", "trade_count", "sharpe",
                     "calmar", "years"))
    )
    for pair, s in (stats.get("per_pair") or {}).items():
        lines.append(
            f"  {pair}: return={_fmt(s.get('total_return'))} pf={_fmt(s.get('profit_factor'))} "
            f"dd={_fmt(s.get('max_drawdown'))} trades={_fmt(s.get('trade_count'))}"
        )
    return "\n".join(lines)


def phase2_feedback(result: dict) -> str:
    """Phase 2 in full: every period is development data."""
    lines = [f"Phase 2 {'PASSED' if result.get('passed') else 'FAILED'} (gate: {PHASE2_GATE})."]
    for w in result.get("windows", []):
        label = w.get("label", "?")
        status = "passed" if w.get("passed") else "failed"
        where = f"{label} ({w['timerange']})" if w.get("timerange") else label
        if "error" in w:
            lines.append(f"  {where}: {status} — error: {_trim(str(w['error']), 600)}")
        else:
            lines.append(
                f"  {where}: {status} (trades={_fmt(w.get('trades'))}, profit_total={_fmt(w.get('profit_total'))}, "
                f"pf={_fmt(w.get('profit_factor'))}, dd={_fmt(w.get('max_drawdown'))})"
            )
    return "\n".join(lines)
