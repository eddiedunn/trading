"""Prompts for the strategy agent. The system prompt is fixed per run so it
caches; the seed and feedback go in user turns."""

from pathlib import Path

from agent.backtest_client import PHASE1_GATE, PHASE2_GATE
from agent.validate import ALLOWED_IMPORTS, EXEMPT_CLASS_ATTRS, MAX_PARAMS

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
- The most recent 6 months are held back. Phase 1 and Phase 2 never use them and you will never
  see any result from them. After a strategy passes Phase 2, a human may run it once on that
  held-back data as the final test; that result decides whether it goes further.
- Columns in the `df` passed to `generate_signals`: timestamp, open, high, low, close, volume.
  Freqtrade's `dataframe` names the time column `date` instead, so keep the logic to the price and
  volume columns and both phases see the same thing.
- Phase 1 charges a 0.045% taker fee per unit of position change (a long-to-short flip pays twice)
  and real Hyperliquid funding on held positions (longs pay positive funding, shorts receive it).
- Phase 2 is Freqtrade using your class's own `stoploss`, trailing stop and `minimal_roi`; Phase 1
  has no stops, so any stop you set makes the phases disagree. Keep stops wide (a disaster stop)
  so both phases trade the same logic, and remember they count toward the numeric-literal cap.

## File format (one file serves every phase)
- A module-level `generate_signals(df) -> pd.Series` returning a position per bar: 1 long, 0 flat,
  -1 short. Phase 1 shifts the series by one bar itself, so compute signals from the current bar's
  values and never look ahead.
- A class named exactly as the strategy, deriving from `IStrategy`, with `populate_indicators`,
  `populate_entry_trend` and `populate_exit_trend`, timeframe "4h", INTERFACE_VERSION = 3.
  It must express the same entry and exit logic as `generate_signals`.
- Import Freqtrade inside try/except ImportError exactly as in the example (Phase 1 has no Freqtrade).
- Imports allowed: {sorted(ALLOWED_IMPORTS)}. No file, network, subprocess or os access.
- At most {MAX_PARAMS} distinct tunable numbers in the whole file. Every numeric literal counts wherever it
  appears: module constants (any case, annotated, tuple-unpacked, inside lists or tuples), arithmetic such as
  `3*4`, and numbers written inline in functions and calls such as `rolling(20)`, `> 1.5` or
  `ewm(span=200)`. A minus sign is part of the value. The same value used twice counts once. Exempt: 0, 1,
  -1, and the values of these IStrategy class attributes: {", ".join(sorted(EXEMPT_CLASS_ATTRS))}.
  `minimal_roi` values do count. Put your knobs in UPPER_CASE module constants and reuse them.
- Look-ahead is rejected before backtesting. Banned: `shift`, `diff` or `pct_change` with a negative
  period, or with a period that is not a plain non-negative number or a module constant holding one;
  `rolling(..., center=True)`; `bfill()`, `backfill()` and `fillna(method="bfill")` (use `ffill()`).
  Anything that uses the whole series at once, such as dividing by the column's overall max or mean,
  or reading `iloc[-1]`, also looks ahead and is not allowed.
- Shorts are optional; if you use them, set `can_short = True` and populate enter_short/exit_short.

## Gates
- Phase 1 (fast filter over the development history): {PHASE1_GATE}.
- The Sharpe bar rises with every attempt in the campaign: each distinct file that reaches Phase 1,
  from any strategy, counts, and a file that fails Phase 1 still counts. Many small tweaks spend the budget
  and raise the bar for everything after them; a few well-reasoned attempts get further.
- A strategy must beat buy-and-hold on Sharpe. Holding every pair equally is the benchmark, and
  each result shows its Sharpe next to yours plus your beta to it. With a beta near 1 the result is
  mostly market exposure; the edge is whatever buy-and-hold does not already give.
- Phase 2 (Freqtrade over three consecutive development periods): {PHASE2_GATE}.
  You see every period's metrics.

## How to work
- Start from the hypothesis you are given and make it concrete. Explain in the module docstring
  what edge the strategy is trying to capture and why it should hold across all three pairs.
- Each submission is an attempt that raises the bar. Think the idea through before replying rather
  than probing the backtest with variations.
- Do not tune to the reported numbers. When a result comes back, change the idea or its structure
  (filter, exit rule, holding logic), not just the constants, and do not add per-pair special cases.
  A strategy that fits the past by having many knobs will fail the held-back data.
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
