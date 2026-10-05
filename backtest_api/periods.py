"""The fixed split between development data and the held-back final-test data.

Everything the agent can learn from (Phase 1, Phase 2) uses bars before
holdout_start(). The final test uses bars from holdout_start() onward, once
per strategy. The date lives in config/holdout.json so it is pinned per
campaign rather than sliding with today's date.
"""

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent


def holdout_config_path() -> Path:
    config_dir = Path(os.environ.get("TRADING_CONFIG_DIR", str(_REPO_ROOT / "config")))
    return config_dir / "holdout.json"


def holdout_start() -> date:
    return date.fromisoformat(json.loads(holdout_config_path().read_text())["holdout_start"])


def campaign_id() -> str:
    """Attempts and final-test results are counted per campaign (one holdout date)."""
    return holdout_start().isoformat()


def development_part(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Rows strictly before the holdout start, index preserved."""
    cutoff = pd.Timestamp(holdout_start(), tz="UTC")
    ts = pd.to_datetime(df[ts_col], utc=True)
    return df[ts < cutoff]


def holdout_part(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Rows at or after the holdout start, index preserved."""
    cutoff = pd.Timestamp(holdout_start(), tz="UTC")
    ts = pd.to_datetime(df[ts_col], utc=True)
    return df[ts >= cutoff]
