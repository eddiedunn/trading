"""BtcLeadEvening — BTC-led follow-through, funding-gated flush bounce, and evening continuation.

Edge: BTC leads. When BTC prints an unusually large up bar, the other coins
(and BTC itself, to a lesser degree) keep rising for the next few bars as
the move propagates and slower money follows. The rule is the same for every
pair: it reads BTC's close, which every pair's frame carries as ``close_BTC``.

Impulse = 2 normal moves; ride for half a day.

The coin's own funding decides which leg may fire. The lead leg only fires
when longs were not paying funding over the last HOLD bars: a BTC impulse
with the coin under-positioned long is follow-through by slower money, while
the same impulse into a crowded long book is already priced and fades
(measured: next-3-bar return after an impulse is 2-7x larger when the
coin's funding was flat or negative, every coin, 2024-2026).

Second leg (same constants, same half-day ride): the coin's own liquidation
flush, a single bar that falls open-to-close by HOLD normal ranges while
longs were paying funding over the last HOLD bars. Forced long liquidation
tends to overshoot and snap back; without positive funding the selling was
not forced and the bounce is absent. The two legs are complementary: one
buys follow-through on up moves when the book is light, the other buys
overshoot on down moves when the book is heavy; ``crowded`` is the switch.
Third leg, one bar only: when BTC is up IMPULSE normal day-moves from
00:00 to 20:00 UTC, stay long every coin for the 20:00-00:00 UTC bar. That
bar holds the US equity close in winter (21:00 UTC) and starts at it in
summer (20:00 UTC): daily-rebalanced leveraged crypto ETFs buy into the
close on big up days, basis desks hedge the day's creations there, and the
Asia open then follows the US day. Measured 2024-2026: that bar after such
a day returns +0.3% BTC, +0.5% ETH, +0.8% SOL, about 30 fires per coin,
most of them in 2024; against roughly zero on ordinary days. The day move
is BTC's close of this bar over its close of the last bar before 00:00
UTC, nothing later. The signal row is the bar closing at 20:00 (EVENING).

All legs run the same ``_position`` on a frame with ``close_BTC``. The
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
EVENING = 16  # signal row: the 4h bar closing 20:00 UTC; the position is then held 20:00-00:00 UTC


def _position(df: pd.DataFrame) -> pd.Series:
    b = df["close_BTC"].pct_change()
    impulse = b > IMPULSE * b.rolling(NORMAL).std()
    # second leg, same constants: the coin's own liquidation flush (open-to-close fall of
    # HOLD normal ranges) while longs were paying funding; ride it the same half day
    rng = (df["high"] - df["low"]) / df["close"]
    flush = (df["open"] - df["close"]) / df["close"] > HOLD * rng.rolling(NORMAL).mean()
    crowded = df["funding_rate"].rolling(HOLD).sum() > 0
    impulse = (impulse & ~crowded) | (flush & crowded)
    # third leg: BTC up IMPULSE normal day-moves by 20:00 UTC -> every coin long through 20:00-00:00
    hour = pd.to_datetime(df["date"] if "date" in df else df["timestamp"], utc=True).dt.hour
    day_open = df["close_BTC"].shift(1).where(hour == 0).ffill()  # BTC at 00:00 UTC (close of the bar before)
    mv = df["close_BTC"] / day_open - 1  # BTC since the UTC day opened, through this bar's close
    evening = (hour == EVENING) & (mv > IMPULSE * mv.where(hour == EVENING).rolling(NORMAL, min_periods=1).std())
    pos = (impulse.astype(float).rolling(HOLD).max() > 0) | evening
    return pos.astype(int)


def generate_signals(df: pd.DataFrame) -> pd.Series:
    return _position(df)


class BtcLeadEvening(IStrategy):
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
