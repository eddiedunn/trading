"""Candidate — the one file the research loop edits (see research/program.md).

Idea: buy forced selling, at two speeds, after the dust settles.

Two kinds of forced selling, one response. A single 4h bar that falls
open-to-close by RANGE_MULT times the coin's normal range is a liquidation
cascade inside a few hours; a close-to-close fall of more than DIP over DAYS
bars (three days) is the same thing stretched over days. In both cases the
sellers were forced, not persuaded, and the next day recovers part of the
move. Measured on development data, each trigger alone has a positive next-day
return on BTC, ETH and SOL in every Phase 2 period (research/scratch/m11,
m19); together they give plenty of trades.

Timing differs. After a one-bar flush the bounce starts at once, so that leg
is held for HOLD bars (one day) from the flush. After a three-day fall the
cascade is still running for a few hours (ETH on average keeps falling for the
first three bars, research/scratch/m21), so that leg waits DELAY bars (half a
day) after the trigger before holding the rest of the day. The three-day
trigger fires only on the bar the fall first exceeds DIP, so a long slide is
bought once, not continuously. Fading upward spikes was measured and loses:
long-only, flat most of the time.

No stop and no ROI: Phase 1 has neither, so Phase 2 trades the same logic.
Both halves run the same ``_position`` on open, high, low and close.
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
NORMAL = 90  # 15 days: window for the normal 4h range
RANGE_MULT = 3  # a flush bar falls open-to-close by this many normal ranges
DAYS = 18  # three days of 4h bars
DIP = 0.10  # a three-day fall this large is a crash
HOLD = 6  # one day
DELAY = 3  # half a day: the wait after a three-day-fall trigger

def _position(df: pd.DataFrame) -> pd.Series:
    cond = df["close"] / df["close"].shift(DAYS) - 1 < -DIP
    trig = cond & ~cond.shift(1, fill_value=False)
    late = trig.shift(DELAY, fill_value=False)
    rng = (df["high"] - df["low"]) / df["close"]
    flush = (df["open"] - df["close"]) / df["close"] > RANGE_MULT * rng.rolling(NORMAL).mean()
    dipleg = late.astype(float).rolling(HOLD - DELAY).max() > 0
    flushleg = flush.astype(float).rolling(HOLD).max() > 0
    return (dipleg | flushleg).astype(int)

def generate_signals(df): return _position(df)

try:
    from freqtrade.strategy import IStrategy
except ImportError:
    IStrategy = object

class ForcedSellingBounce(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "4h"
    can_short = False
    minimal_roi = {}  # no ROI exits
    stoploss = -1  # no stop: both phases trade the same logic
    trailing_stop = False
    startup_candle_count = NORMAL + HOLD
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["pos"] = _position(dataframe); return dataframe
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["pos"] == 1, "enter_long"] = 1; return dataframe
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["pos"] != 1, "exit_long"] = 1; return dataframe
