"""Prompts for the strategy agent. The system prompt is fixed per run so it
caches; the seed and feedback go in user turns."""

from pathlib import Path

from agent.backtest_client import PHASE1_GATE, PHASE2_GATE
from agent.validate import ALLOWED_IMPORTS, EXEMPT_CLASS_ATTRS, MAX_PARAMS

_REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PATH = _REPO_ROOT / "strategies" / "examples" / "EmaCross.py"

# Starting hypotheses for when the caller gives none. Varied on purpose: data most traders
# ignore (funding, the other coins), shorts, and being flat most of the time.
DEFAULT_SEEDS = [
    "Funding-extreme fade: when this coin's trailing funding (funding_rate summed over a few days) is "
    "far above its own recent normal, longs are crowded and paying to stay in, so go short; go long "
    "when it is far below normal; be flat otherwise. Exit when funding returns to normal.",
    "Funding carry with the trend: hold the side that receives funding (short when trailing funding is "
    "clearly positive, long when clearly negative), but only while the price trend agrees with that "
    "side. Flat when they disagree, so you are paid to hold a position the market is not fighting.",
    "Relative-strength rotation across BTC, ETH and SOL using close_BTC/close_ETH/close_SOL: long this "
    "coin only while its multi-day return is the best of the three and positive, short it only while "
    "it is the worst and negative, flat in between.",
    "Lead-lag: BTC tends to move first and ETH/SOL follow. Trade this coin off a sharp recent move in "
    "close_BTC that this coin has not matched yet; on the BTC pair itself either stay flat or use "
    "BTC's own trend, since there is no leader to follow.",
    "Volatility regime switch: measure realised volatility (or ATR as a fraction of price) against its "
    "own recent history; in high-volatility regimes follow the trend in either direction, in quiet "
    "choppy regimes stay flat.",
    "Short-side breakdowns: go short when price closes below its recent N-bar low while BTC is below its "
    "long average (a broad downtrend); go flat when price reclaims a shorter channel. Never long.",
    "Market-neutral pair on the ETH/BTC ratio (close_ETH / close_BTC): when the ratio is stretched far "
    "from its rolling mean, the ETH pair takes one side and the BTC pair the opposite side on the same "
    "bar, and both go flat when the ratio reverts. Each pair computes the same ratio from the shared "
    "columns and returns only its own leg; SOL stays flat.",
    "Mean reversion after liquidation-style spikes: a bar whose range is several times the recent "
    "average range and whose volume is well above its average marks forced selling or buying; fade "
    "that bar's direction for a short, fixed holding period, otherwise flat.",
    "Funding divergence across coins: compare funding_BTC, funding_ETH and funding_SOL over the last "
    "few days. Short the coin whose funding is far richer than the other two (its longs are the most "
    "crowded); long the one whose funding is far cheaper; flat when they agree.",
    "Uncrowded breakouts: take a channel breakout in either direction only when trailing funding is not "
    "already stretched in that direction (longs not paying up for an upside break, shorts not paying "
    "up for a downside one); exit on a close back inside a shorter channel.",
]

FREQTRADE_SNIPPET = '''\
    def informative_pairs(self):
        coins = [(f"{c}/USDC:USDC", self.timeframe) for c in ("BTC", "ETH", "SOL")]
        funding = [(f"{c}/USDC:USDC", "1h", "funding_rate") for c in ("BTC", "ETH", "SOL")]
        return coins + funding

    def populate_indicators(self, dataframe, metadata):
        # this pair's funding_rate: hourly rates paid in (T, T+4h] summed onto the bar opening at T
        fr = self.dp.get_pair_dataframe(metadata["pair"], "1h", candle_type="funding_rate")
        dataframe["funding_rate"] = np.nan
        if not fr.empty:
            per_bar = fr.set_index("date")["open"].resample(self.timeframe, closed="right", label="left").sum()
            dataframe["funding_rate"] = per_bar.reindex(pd.DatetimeIndex(dataframe["date"])).to_numpy()
        # close_BTC / close_ETH / close_SOL: same 4h timeframe, bar T matched to bar T
        for coin in ("BTC", "ETH", "SOL"):
            inf = self.dp.get_pair_dataframe(f"{coin}/USDC:USDC", self.timeframe)[["date", "close"]]
            dataframe = merge_informative_pair(dataframe, inf, self.timeframe, self.timeframe,
                                               ffill=False, append_timeframe=False, suffix=coin)
        dataframe["pos"] = _position(dataframe)  # the same function generate_signals calls
        return dataframe
'''


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
- `generate_signals` is called once per pair. Columns in its `df`:
  - timestamp, open, high, low, close, volume: this pair's 4h bar (timestamp is the bar's open time).
  - funding_rate: this pair's signed funding summed over the bar's own four hours, i.e. the hourly
    payments stamped T+1h .. T+4h for the bar opening at T. The last one is paid at the bar's close,
    so all of it is known when the bar closes and you may use it in that bar's signal. It is the
    same funding a position held over the bar pays. Positive means longs pay shorts. Typical size is
    about 0.00005 per bar (0.00125% an hour, roughly 11% a year for longs); it spikes in crowded
    markets and goes negative when shorts are crowded. NaN where there is no funding data (e.g. the
    first bars), so guard rolling sums against NaN and treat NaN as "no signal".
  - close_BTC, close_ETH, close_SOL: each coin's close for the bar with the same open time (on the
    BTC pair, close_BTC equals close). NaN where that coin has no bar.
  - funding_BTC, funding_ETH, funding_SOL: each coin's funding_rate, aligned the same way.
  Every value in row t is known at the close of bar t; nothing comes from a later bar.
- Every pair's df carries the same close_*/funding_* columns, so the pairs can act together: each
  pair computes the same cross-coin signal and returns only its own leg. That is the only way to
  express "long the strongest, short the weakest" or a market-neutral pair: there is no portfolio
  call, each pair gets its own equal slot, and a pair cannot size itself against another.
- Freqtrade's `dataframe` names the time column `date` and has only the OHLCV columns; your class
  must build funding_rate and the close_*/funding_* columns itself (see Freqtrade parity below).
- Phase 1 charges a 0.045% taker fee per unit of position change (a long-to-short flip pays twice)
  and real Hyperliquid funding on held positions (longs pay positive funding, shorts receive it).
- Phase 2 is Freqtrade using your class's own `stoploss`, trailing stop and `minimal_roi`; Phase 1
  has no stops, so any stop you set makes the phases disagree. Keep stops wide (a disaster stop)
  so both phases trade the same logic, and remember they count toward the numeric-literal cap.
- Phase 2 gives indicators 20 days (120 candles) of warm-up before its first period; keep
  `startup_candle_count` at 120 or less or the first period fails its range check.

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
- Shorts are allowed and often needed: return -1 in `generate_signals`, and in the class set
  `can_short = True` and populate enter_short/exit_short.

## Freqtrade parity
Put the logic in one module-level function, say `_position(df)`, that reads only columns both phases
have (close, funding_rate, close_BTC, ...; never timestamp or date), and call it from both
`generate_signals` and `populate_indicators`. Then derive the class's signals from that position as
levels: enter_long where pos == 1, exit_long where pos != 1, enter_short where pos == -1, exit_short
where pos != -1. Import `merge_informative_pair` together with `IStrategy` inside the same try/except.
Build the extra columns like this (`self.dp.get_pair_dataframe(pair, "1h", candle_type="funding_rate")`
returns the hourly funding with the rate in `open`; `informative_pairs` keeps the data fresh when the
strategy later runs live):
```python
{FREQTRADE_SNIPPET}```
For funding_BTC/ETH/SOL, run the same funding block with `f"{{coin}}/USDC:USDC"` instead of
`metadata["pair"]` and store it as `funding_{{coin}}`. Strings such as "1h" are not numeric literals.

## Gates
- Phase 1 (fast filter over the development history): {PHASE1_GATE}.
- Every file that reaches Phase 1 is counted. The held-back final test raises its Sharpe bar for
  every strategy the campaign sends to it, and strategies tuned to development data rarely survive
  it, so prefer a few well-reasoned ideas over many small tweaks.
- A strategy must beat buy-and-hold on Sharpe. Holding every pair equally is the benchmark, and
  each result shows its Sharpe next to yours plus your beta to it. With a beta near 1 the result is
  mostly market exposure; the edge is whatever buy-and-hold does not already give.
- Phase 2 (Freqtrade over three consecutive development periods): {PHASE2_GATE}.
  You see every period's metrics.

## Why always-long fails here
- Every pair must keep its own max drawdown above -35% and the combined account above -20% over the
  development history, which includes crypto drawdowns far deeper than that. A strategy that is long
  most of the time inherits them and fails. Being flat, short or hedged much of the time is fine and
  expected; a strategy that only trades when its edge is present is what passes.

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
