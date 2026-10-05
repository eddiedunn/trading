"""RelativeStrength — example that uses the other coins' closes.

Edge: among BTC, ETH and SOL, the coin with the best recent return tends to keep
leading for a while (cross-sectional momentum), and the weakest tends to keep
lagging. Each pair is traded on its own, so the rule is written per pair:

- long this coin while its LOOKBACK-bar return is the best of the three and positive;
- short it while its return is the worst of the three and negative;
- flat otherwise.

Across the three pairs this is close to "long the leader, short the laggard" when
the market trends, and mostly flat when the coins move together. It cannot size
the legs against each other: every pair has its own equal slot.

Both halves run the same ``_position`` on a frame with ``close``, ``close_BTC``,
``close_ETH`` and ``close_SOL``. Phase 1 passes those columns in ``df``; the
Freqtrade class merges them in with ``informative_pairs`` and
``merge_informative_pair`` (same 4h timeframe, so bar T is matched to bar T).
"""

import pandas as pd
from pandas import DataFrame

LOOKBACK = 42  # 7 days of 4h bars
COINS = ("BTC", "ETH", "SOL")


def _position(df: pd.DataFrame) -> pd.Series:
    """1 long, -1 short, 0 flat for each bar, from the closes only."""
    own = df["close"].pct_change(LOOKBACK, fill_method=None)
    others = pd.concat([df[f"close_{c}"].pct_change(LOOKBACK, fill_method=None) for c in COINS], axis=1)
    best, worst = others.max(axis=1, skipna=False), others.min(axis=1, skipna=False)
    pos = pd.Series(0, index=df.index)
    pos[(own >= best) & (own > 0)] = 1
    pos[(own <= worst) & (own < 0)] = -1
    return pos


def generate_signals(df: pd.DataFrame) -> pd.Series:
    return _position(df)


try:
    from freqtrade.strategy import IStrategy, merge_informative_pair
except ImportError:  # Phase 1 runs without Freqtrade installed
    IStrategy = object


class RelativeStrength(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "4h"
    can_short = True
    minimal_roi = {"0": 100}  # unreachable: ROI never forces an exit
    stoploss = -0.15  # disaster stop only, so Phase 1 and Phase 2 trade the same logic
    trailing_stop = False
    startup_candle_count = LOOKBACK

    def informative_pairs(self):
        return [(f"{c}/USDC:USDC", self.timeframe) for c in COINS]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for coin in COINS:
            inf = self.dp.get_pair_dataframe(f"{coin}/USDC:USDC", self.timeframe)[["date", "close"]]
            dataframe = merge_informative_pair(dataframe, inf, self.timeframe, self.timeframe,
                                               ffill=False, append_timeframe=False, suffix=coin)
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
