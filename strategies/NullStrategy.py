"""
NullStrategy — placeholder that never opens positions.
Used as the default strategy until a real strategy is promoted to live.
"""

from freqtrade.strategy import IStrategy
import pandas as pd


class NullStrategy(IStrategy):
    """
    Does nothing — never buys or sells.
    Serves as a placeholder so Freqtrade starts and the REST API is available.
    Replace with a promoted strategy via: trading_client.py live promote --strategy <name>
    """

    INTERFACE_VERSION = 3

    minimal_roi = {"0": 100}  # unreachably high — never triggers
    stoploss = -0.99           # never triggers
    timeframe = "4h"

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe
