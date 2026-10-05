"""Prompts for the strategy agent. The system prompt is fixed per run so it
caches; the seed and feedback go in user turns."""

from pathlib import Path

from agent.backtest_client import PHASE1_GATE, PHASE2_GATE
from agent.validate import ALLOWED_IMPORTS, MAX_PARAMS

_REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PATH = _REPO_ROOT / "strategies" / "examples" / "EmaCross.py"

# A few starting hypotheses for when the caller gives none.
DEFAULT_SEEDS = [
    "Trend following with a volatility filter: only hold a long when a slow trend measure is up "
    "and recent realised volatility is not extreme.",
    "Donchian-style breakout: enter on a close above the recent N-bar high, exit on a close below "
    "the recent M-bar low, with M shorter than N.",
    "Mean reversion inside a trend: buy pullbacks to a moving average while a longer average is rising, "
    "exit when price closes back above a short-term band.",
    "Momentum with a regime filter: hold when the N-bar return is positive and the bar range (ATR as a "
    "fraction of price) is below its recent median.",
    "Volume-confirmed breakout: a close above a channel only counts when volume is above its rolling average.",
]


def system_prompt() -> str:
    example = EXAMPLE_PATH.read_text()
    return f"""You write candidate trading strategies for Hyperliquid perpetual futures. Each reply is one
complete Python file that will be backtested automatically. You cannot run code yourself; the
harness runs it and reports back.

## Data and venue
- Pairs: BTC, ETH and SOL USDC perpetuals on Hyperliquid, 4h candles, history from about 2023-12.
- Columns available in `df`: date, open, high, low, close, volume.
- Phase 1 charges a 0.045% taker fee per position change and a 0.01%/bar funding drag while in a position.
- Phase 2 is Freqtrade with a 5% stoploss and a 2% trailing stop from config; Phase 1 has neither,
  so the two phases can disagree on the same logic. Keep the logic simple enough that both agree.

## File format (one file serves every phase)
- A module-level `generate_signals(df) -> pd.Series` returning a position per bar: 1 long, 0 flat,
  -1 short. Phase 1 shifts the series by one bar itself, so compute signals from the current bar's
  values; never use shift with a negative period or any other look-ahead.
- A class named exactly as the strategy, deriving from `IStrategy`, with `populate_indicators`,
  `populate_entry_trend` and `populate_exit_trend`, timeframe "4h", INTERFACE_VERSION = 3.
  It must express the same entry and exit logic as `generate_signals`.
- Import Freqtrade inside try/except ImportError exactly as in the example (Phase 1 has no Freqtrade).
- Imports allowed: {sorted(ALLOWED_IMPORTS)}. No file, network, subprocess or os access.
- At most {MAX_PARAMS} module-level numeric constants (UPPER_CASE = number). Those are your knobs.
- Shorts are optional; if you use them, set `can_short = True` and populate enter_short/exit_short.

## Gates
- Phase 1 (fast filter over the whole history): {PHASE1_GATE}.
- Phase 2 (Freqtrade walk-forward over three consecutive windows): {PHASE2_GATE}.
  You will see metrics for the earlier windows and only pass/fail for the final out-of-sample window.

## How to work
- Start from the hypothesis you are given and make it concrete. Explain in the module docstring
  what edge the strategy is trying to capture and why it should hold across all three pairs.
- Do not tune to the reported numbers. When a result comes back, change the idea or its structure
  (filter, exit rule, holding logic), not just the constants, and do not add per-pair special cases.
  A strategy that fits the past by having many knobs will fail the later windows.
- Prefer round, conventional parameter values. Fewer trades with a clearer edge beat many marginal ones,
  but the Phase 1 gate needs more than 30 trades across the three pairs combined.
- Keep the class name the same across revisions of the same strategy.

## Reply format
Reply with exactly one ```python fenced block containing the whole file and nothing else.

## Example (the canonical format; its logic is a placeholder, not a recommendation)
```python
{example}```
"""


def initial_prompt(seed: str, name: str) -> str:
    return (
        f"Write a strategy named `{name}` (file name, class name and strategy name are all `{name}`).\n\n"
        f"Hypothesis to build on:\n{seed}\n\n"
        "Reply with the complete file."
    )


def revision_prompt(feedback: str) -> str:
    return (
        f"Result of the last submission:\n\n{feedback}\n\n"
        "Revise the strategy. Remember: change the logic or structure rather than fitting the constants to "
        "these numbers, keep the class name, and reply with the complete revised file."
    )
