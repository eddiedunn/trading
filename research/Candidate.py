"""Candidate — the one file the research loop edits (see research/program.md).

Starting point: the FundingFade example.

Edge: when perpetual funding is far above its recent normal, longs are crowded
and paying heavily to stay in; those positions tend to unwind, so price tends to
mean-revert down. Very negative funding is the mirror image. The strategy shorts
when trailing funding is extreme-positive, goes long when it is extreme-negative,
and is flat otherwise. It also collects the funding it fades while it holds
(shorts receive positive funding). Being flat most of the time is intended.

"Extreme" is a z-score of the trailing TRAIL-bar funding sum against its own
LOOKBACK-bar history, so the same rule fits every coin. A position closes when
that z-score crosses back through zero (funding back to normal).

Both halves run the same ``_position`` on a frame with ``close`` and
``funding_rate``:

- Phase 1 passes ``funding_rate`` in ``df``: the funding paid over each bar's own
  hours (stamps T+1h .. T+4h for the bar opening at T), all paid by the bar's close.
- The Freqtrade class builds the same column from
  ``self.dp.get_pair_dataframe(pair, "1h", candle_type="funding_rate")``, summing
  the hourly rates into 4h bins that are closed on the right and labelled by
  their left edge, i.e. (T, T+4h] goes to the bar opening at T.
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

TRAIL = 18  # 3 days of 4h bars
LOOKBACK = 90  # 15 days; TRAIL + LOOKBACK stays under Phase 2's 120-candle warm-up
ENTRY_Z = 2


def _position(df: pd.DataFrame) -> pd.Series:
    """1 long, -1 short, 0 flat for each bar, from close and funding_rate only."""
    trailing = df["funding_rate"].rolling(TRAIL).sum()
    z = (trailing - trailing.rolling(LOOKBACK).mean()) / trailing.rolling(LOOKBACK).std()
    pos = pd.Series(np.nan, index=df.index)
    pos[np.sign(z) != np.sign(z.shift(1))] = 0  # z crossed zero, or no data: flat
    pos[z > ENTRY_Z] = -1  # crowded longs paying up: fade them
    pos[z < -ENTRY_Z] = 1  # crowded shorts paying up: fade them
    return pos.ffill().fillna(0).astype(int)


def generate_signals(df: pd.DataFrame) -> pd.Series:
    return _position(df)


try:
    from freqtrade.strategy import IStrategy
except ImportError:  # Phase 1 runs without Freqtrade installed
    IStrategy = object


class Candidate(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "4h"
    can_short = True
    minimal_roi = {"0": 100}  # unreachable: ROI never forces an exit
    stoploss = -0.15  # disaster stop only, so Phase 1 and Phase 2 trade the same logic
    trailing_stop = False
    startup_candle_count = TRAIL + LOOKBACK

    def informative_pairs(self):
        # Needed for dry-run/live so the bot keeps the hourly funding fresh.
        return [(pair, "1h", "funding_rate") for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        fr = self.dp.get_pair_dataframe(metadata["pair"], "1h", candle_type="funding_rate")
        dataframe["funding_rate"] = np.nan
        if not fr.empty:
            # Sum the hourly rates paid in (T, T+4h] onto the bar opening at T.
            per_bar = fr.set_index("date")["open"].resample(self.timeframe, closed="right", label="left").sum()
            dataframe["funding_rate"] = per_bar.reindex(pd.DatetimeIndex(dataframe["date"])).to_numpy()
        dataframe["pos"] = _position(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["pos"] == 1, "enter_long"] = 1
        dataframe.loc[dataframe["pos"] == -1, "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["pos"] != 1, "exit_long"] = 1
        dataframe.loc[dataframe["pos"] != -1, "exit_short"] = 1
        return dataframe
