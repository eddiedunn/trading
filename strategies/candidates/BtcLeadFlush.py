"""Candidate — the one file the research loop edits (see research/program.md).

Edge: BTC leads. When BTC prints an unusually large up bar, the other coins
(and BTC itself, to a lesser degree) keep rising for the next few bars as
the move propagates and slower money follows. The rule is the same for every
pair: it reads BTC's close, which every pair's frame carries as ``close_BTC``.

Impulse = 2 normal moves; ride for half a day.

Second leg (same constants, same half-day ride): the coin's own liquidation
flush, a single bar that falls open-to-close by HOLD normal ranges while
longs were paying funding over the last HOLD bars. Forced long liquidation
tends to overshoot and snap back; without positive funding the selling was
not forced and the bounce is absent. The two legs are complementary: one
buys follow-through on up moves, the other buys overshoot on down moves.
Both halves run the same ``_position`` on a frame with ``close_BTC``. The
Freqtrade class builds that column with ``merge_informative_pair`` from BTC's
own 4h candles (bar T matched to bar T, no forward fill), the same alignment
Phase 1 uses, so there is no look-ahead.
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
try:
    from freqtrade.strategy import IStrategy, merge_informative_pair
except ImportError:
    IStrategy = object

NORMAL = 90  # 15 days: window for BTC's normal 4h move
IMPULSE = 2  # a BTC bar this many normal moves up is an impulse
HOLD = 3  # bars to ride the follow-through


def _position(df: pd.DataFrame) -> pd.Series:
    b = df["close_BTC"].pct_change()
    impulse = b > IMPULSE * b.rolling(NORMAL).std()
    # second leg, same constants: the coin's own liquidation flush (open-to-close fall of
    # HOLD normal ranges) while longs were paying funding; ride it the same half day
    rng = (df["high"] - df["low"]) / df["close"]
    flush = (df["open"] - df["close"]) / df["close"] > HOLD * rng.rolling(NORMAL).mean()
    crowded = df["funding_rate"].rolling(HOLD).sum() > 0
    impulse = impulse | (flush & crowded)
    return (impulse.astype(float).rolling(HOLD).max() > 0).astype(int)


def generate_signals(df: pd.DataFrame) -> pd.Series:
    return _position(df)


class BtcLeadFlush(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "4h"
    can_short = False
    minimal_roi = {}
    stoploss = -1
    trailing_stop = False
    startup_candle_count = NORMAL + HOLD

    def informative_pairs(self):
        coins = [("BTC/USDC:USDC", self.timeframe)]
        funding = [(f"{c}/USDC:USDC", "1h", "funding_rate") for c in ("BTC", "ETH", "SOL")]
        return coins + funding

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # this pair's funding_rate: hourly rates paid in (T, T+4h] summed onto the bar opening at T
        fr = self.dp.get_pair_dataframe(metadata["pair"], "1h", candle_type="funding_rate")
        dataframe["funding_rate"] = np.nan
        if not fr.empty:
            per_bar = fr.set_index("date")["open"].resample(self.timeframe, closed="right", label="left").sum()
            dataframe["funding_rate"] = per_bar.reindex(pd.DatetimeIndex(dataframe["date"])).to_numpy()
        # close_BTC: BTC's close for the bar with the same open time (bar T matched to bar T)
        inf = self.dp.get_pair_dataframe("BTC/USDC:USDC", self.timeframe)[["date", "close"]]
        dataframe = merge_informative_pair(dataframe, inf, self.timeframe, self.timeframe,
                                           ffill=False, append_timeframe=False, suffix="BTC")
        dataframe["pos"] = _position(dataframe)  # the same function generate_signals calls
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["pos"] == 1, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["pos"] != 1, "exit_long"] = 1
        return dataframe
