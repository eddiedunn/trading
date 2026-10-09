"""The Freqtrade configs must not set stops or ROI.

Freqtrade lets config values override strategy class attributes, so a stop in
a config silently replaces every strategy's own exits. Each strategy declares
its own; the configs stay out of it.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

from paper.orchestrator import _build_paper_config

ROOT = Path(__file__).resolve().parent.parent

STRATEGY_EXIT_KEYS = {
    "stoploss",
    "trailing_stop",
    "trailing_stop_positive",
    "trailing_stop_positive_offset",
    "trailing_only_offset_is_reached",
    "minimal_roi",
}


def test_backtest_config_sets_no_exits():
    cfg = json.loads((ROOT / "config" / "backtest.json").read_text())
    assert STRATEGY_EXIT_KEYS.isdisjoint(cfg)


@patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
def test_paper_config_sets_no_exits():
    assert STRATEGY_EXIT_KEYS.isdisjoint(_build_paper_config("S", 8090, "paper_s"))


def test_live_template_sets_no_exits():
    # No Jinja/Ansible in the test env, so check the template text for the keys.
    text = (ROOT / "deploy/ansible/roles/trading_live/templates/live-config.json.j2").read_text()
    assert [k for k in STRATEGY_EXIT_KEYS if f'"{k}"' in text] == []


def test_strategies_declare_their_own_stoploss():
    """Freqtrade refuses a strategy with no stoploss once the config stops supplying one."""
    for path in [ROOT / "strategies/examples/EmaCross.py", ROOT / "strategies/NullStrategy.py"]:
        text = path.read_text()
        assert "    stoploss = " in text, path
        assert "    minimal_roi = " in text, path


def test_backtest_config_splits_balance_across_every_pair():
    """"unlimited" stakes balance / (max_open_trades - open trades), so every scored pair can open."""
    from backtest_api.universe import COIN_PAIRS
    cfg = json.loads((ROOT / "config" / "backtest.json").read_text())
    assert cfg["stake_amount"] == "unlimited"
    assert cfg["max_open_trades"] == len(COIN_PAIRS)
    assert cfg["exchange"]["pair_whitelist"] == [f"{p.split('_')[0]}/USDC:USDC" for p in COIN_PAIRS]
    assert cfg["dry_run_wallet"] >= 1000
