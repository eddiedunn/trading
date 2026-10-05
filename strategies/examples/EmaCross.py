"""EmaCross — example candidate showing the two-in-one strategy file format.

One file serves every phase of the pipeline:

- Phase 1 (numpy fast filter) imports the module and calls the module-level
  ``generate_signals(df)``, which returns a 0/1 position series.
- Phases 2-4 (Freqtrade walk-forward, paper arena, live) load the ``IStrategy``
  subclass whose name matches the file name.

Goes long when the fast EMA crosses above the slow EMA and goes flat when it
crosses back below. The entry is the cross itself, not "fast is above slow", so
after a stoploss exit the strategy waits for the next cross instead of buying
again on the next candle.

Stops and ROI live on the class; the Freqtrade configs set neither, so these
values are the ones that apply. Phase 1 has no stoploss.
"""

import pandas as pd
from pandas import DataFrame

FAST = 12
SLOW = 26


def generate_signals(df: pd.DataFrame) -> pd.Series:
    """Phase 1 signal: 1 = long, 0 = flat.

    Long while fast is above slow. With no stoploss in Phase 1 this is the
    position the class holds: it enters on the cross up and exits once fast
    drops below slow.
    """
    fast = df["close"].ewm(span=FAST, adjust=False).mean()
    slow = df["close"].ewm(span=SLOW, adjust=False).mean()
    return (fast > slow).astype(int)


try:
    from freqtrade.strategy import IStrategy
except ImportError:  # Phase 1 runs without Freqtrade installed
    IStrategy = object


class EmaCross(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "4h"
    can_short = False
    minimal_roi = {"0": 100}  # unreachable: ROI never forces an exit
    stoploss = -0.05
    trailing_stop = False
    startup_candle_count = SLOW * 3

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = dataframe["close"].ewm(span=FAST, adjust=False).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(span=SLOW, adjust=False).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        fast, slow = dataframe["ema_fast"], dataframe["ema_slow"]
        crossed_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        dataframe.loc[crossed_up, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe["ema_fast"] < dataframe["ema_slow"], "exit_long"] = 1
        return dataframe
