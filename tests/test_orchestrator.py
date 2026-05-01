"""Unit tests for paper arena orchestrator.

Tests config building, port assignment, container name generation,
and command construction for podman spawn/teardown.
"""

import json
import os
from unittest.mock import patch, MagicMock, call
from pathlib import Path

import pytest

from paper.orchestrator import (
    spawn_paper_instance,
    teardown_paper_instance,
    teardown_all,
    list_paper_instances,
    _build_paper_config,
    PaperInstance,
    BASE_PORT,
    MAX_SLOTS,
    PAPER_CONFIGS_DIR,
)


class TestBuildPaperConfig:
    """Test Freqtrade config generation."""

    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
    def test_config_basic_structure(self):
        """Config has required top-level keys."""
        cfg = _build_paper_config("TestStrat", 8090, "paper_teststrat")

        assert "exchange" in cfg
        assert "trading_mode" in cfg
        assert "stake_currency" in cfg
        assert "dry_run" in cfg
        assert "api_server" in cfg

    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
    def test_config_port_assignment(self):
        """Config uses provided port for API server."""
        port = 8092
        cfg = _build_paper_config("TestStrat", port, "paper_teststrat")

        assert cfg["api_server"]["listen_port"] == port

    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
    def test_config_dry_run_enabled(self):
        """Config has dry_run=True and correct wallet."""
        cfg = _build_paper_config("TestStrat", 8090, "paper_teststrat")

        assert cfg["dry_run"] is True
        assert cfg["dry_run_wallet"] == 100

    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
    def test_config_stake_settings(self):
        """Config has correct stake amount and max trades."""
        cfg = _build_paper_config("TestStrat", 8090, "paper_teststrat")

        assert cfg["stake_amount"] == 33
        assert cfg["max_open_trades"] == 3

    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
    def test_config_pair_whitelist(self):
        """Config has BTC, ETH, SOL pairs."""
        cfg = _build_paper_config("TestStrat", 8090, "paper_teststrat")

        assert "pair_whitelist" in cfg
        pairs = cfg["pair_whitelist"]
        assert len(pairs) == 3
        assert any("BTC" in p for p in pairs)
        assert any("ETH" in p for p in pairs)
        assert any("SOL" in p for p in pairs)

    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
    def test_config_timeframe(self):
        """Config uses 4h timeframe."""
        cfg = _build_paper_config("TestStrat", 8090, "paper_teststrat")

        assert cfg["timeframe"] == "4h"

    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
    def test_config_exchange_settings(self):
        """Config uses hyperliquid futures."""
        cfg = _build_paper_config("TestStrat", 8090, "paper_teststrat")

        assert cfg["exchange"]["name"] == "hyperliquid"
        assert cfg["trading_mode"] == "futures"
        assert cfg["margin_mode"] == "isolated"
        assert cfg["stake_currency"] == "USDC"

    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"}, clear=True)
    def test_config_prod_no_testnet_urls(self):
        """Without TRADING_ENV=testnet, no API URL override is set."""
        os.environ["FREQTRADE_API_PASSWORD"] = "changeme"
        cfg = _build_paper_config("TestStrat", 8090, "paper_teststrat")
        assert "urls" not in cfg["exchange"]["ccxt_config"]

    @patch.dict(
        os.environ,
        {"FREQTRADE_API_PASSWORD": "changeme", "TRADING_ENV": "testnet"},
    )
    def test_config_testnet_overrides_api_url(self):
        """TRADING_ENV=testnet routes to hyperliquid-testnet.xyz."""
        cfg = _build_paper_config("TestStrat", 8090, "paper_teststrat")
        urls = cfg["exchange"]["ccxt_config"]["urls"]["api"]
        assert "testnet" in urls["public"]
        assert "testnet" in urls["private"]

    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
    def test_config_stoploss(self):
        """Config has trailing stop loss settings."""
        cfg = _build_paper_config("TestStrat", 8090, "paper_teststrat")

        assert cfg["stoploss"] == -0.05
        assert cfg["trailing_stop"] is True
        assert cfg["trailing_stop_positive"] == 0.02


class TestPaperInstance:
    """Test PaperInstance dataclass."""

    def test_dataclass_creation(self):
        """Create PaperInstance with all fields."""
        inst = PaperInstance(
            strategy_name="TestStrat",
            port=8090,
            container_name="paper_teststrat_0",
            db_schema="paper_teststrat",
        )

        assert inst.strategy_name == "TestStrat"
        assert inst.port == 8090
        assert inst.container_name == "paper_teststrat_0"
        assert inst.db_schema == "paper_teststrat"

    def test_to_dict(self):
        """to_dict converts dataclass to dict."""
        inst = PaperInstance(
            strategy_name="TestStrat",
            port=8090,
            container_name="paper_teststrat_0",
            db_schema="paper_teststrat",
        )
        d = inst.to_dict()

        assert isinstance(d, dict)
        assert d["strategy_name"] == "TestStrat"
        assert d["port"] == 8090


class TestSpawnPaperInstance:
    """Test spawning paper trading containers."""

    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
    @patch("paper.orchestrator.subprocess.run")
    @patch("paper.orchestrator.Path.mkdir")
    @patch("paper.orchestrator.Path.write_text")
    def test_spawn_valid_slot(self, mock_write, mock_mkdir, mock_subprocess):
        """Spawn instance for valid slot."""
        mock_subprocess.return_value = MagicMock(returncode=0)

        instance = spawn_paper_instance("TestStrat", slot=0)

        assert instance.strategy_name == "TestStrat"
        assert instance.port == BASE_PORT + 0
        assert instance.container_name == "paper_teststrat_0"
        assert instance.db_schema == "paper_teststrat"

    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
    @patch("paper.orchestrator.subprocess.run")
    @patch("paper.orchestrator.Path.mkdir")
    @patch("paper.orchestrator.Path.write_text")
    def test_spawn_different_slots(self, mock_write, mock_mkdir, mock_subprocess):
        """Different slots get different ports."""
        mock_subprocess.return_value = MagicMock(returncode=0)

        inst0 = spawn_paper_instance("Strat", slot=0)
        inst3 = spawn_paper_instance("Strat", slot=3)

        assert inst0.port == BASE_PORT + 0
        assert inst3.port == BASE_PORT + 3
        assert inst0.port != inst3.port

    def test_spawn_invalid_slot(self):
        """Slot >= MAX_SLOTS raises ValueError."""
        with pytest.raises(ValueError, match="exceeds max"):
            spawn_paper_instance("TestStrat", slot=MAX_SLOTS)

    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
    @patch("paper.orchestrator.subprocess.run")
    @patch("paper.orchestrator.Path.mkdir")
    @patch("paper.orchestrator.Path.write_text")
    def test_spawn_writes_config(self, mock_write, mock_mkdir, mock_subprocess):
        """spawn_paper_instance writes config JSON."""
        mock_subprocess.return_value = MagicMock(returncode=0)

        spawn_paper_instance("TestStrat", slot=0)

        # Verify write_text was called with JSON
        mock_write.assert_called_once()
        config_str = mock_write.call_args[0][0]
        # Should be valid JSON
        cfg = json.loads(config_str)
        assert cfg["dry_run"] is True

    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
    @patch("paper.orchestrator.subprocess.run")
    @patch("paper.orchestrator.Path.mkdir")
    @patch("paper.orchestrator.Path.write_text")
    def test_spawn_podman_command(self, mock_write, mock_mkdir, mock_subprocess):
        """Verify podman run command structure."""
        mock_subprocess.return_value = MagicMock(returncode=0)

        spawn_paper_instance("MyStrat", slot=1)

        call_args = mock_subprocess.call_args[0][0]
        assert call_args[0] == "podman"
        assert call_args[1] == "run"
        assert "-d" in call_args
        assert "--name" in call_args
        assert "paper_mystrat_1" in call_args
        assert "-v" in call_args
        assert "-p" in call_args
        assert "freqtradeorg/freqtrade:stable" in call_args
        assert "trade" in call_args
        assert "--strategy" in call_args
        assert "MyStrat" in call_args

    @patch.dict(os.environ, {"FREQTRADE_API_PASSWORD": "changeme"})
    @patch("paper.orchestrator.subprocess.run")
    @patch("paper.orchestrator.Path.mkdir")
    @patch("paper.orchestrator.Path.write_text")
    def test_spawn_port_binding(self, mock_write, mock_mkdir, mock_subprocess):
        """Verify port binding format in podman command."""
        mock_subprocess.return_value = MagicMock(returncode=0)

        spawn_paper_instance("TestStrat", slot=2)

        call_args = mock_subprocess.call_args[0][0]
        port_idx = call_args.index("-p")
        port_binding = call_args[port_idx + 1]
        expected_port = BASE_PORT + 2
        assert str(expected_port) in port_binding


class TestTeardownPaperInstance:
    """Test tearing down paper trading containers."""

    @patch("paper.orchestrator.subprocess.run")
    @patch("paper.orchestrator.Path.unlink")
    def test_teardown_instance(self, mock_unlink, mock_subprocess):
        """teardown_paper_instance stops and removes container."""
        instance = PaperInstance(
            strategy_name="TestStrat",
            port=8090,
            container_name="paper_teststrat_0",
            db_schema="paper_teststrat",
        )

        teardown_paper_instance(instance)

        # Should call podman stop and rm
        calls = mock_subprocess.call_args_list
        assert len(calls) >= 2
        # First call should be podman stop
        assert calls[0][0][0][0] == "podman"
        assert calls[0][0][0][1] == "stop"
        assert "paper_teststrat_0" in calls[0][0][0]
        # Second call should be podman rm
        assert calls[1][0][0][0] == "podman"
        assert calls[1][0][0][1] == "rm"

    @patch("paper.orchestrator.subprocess.run")
    @patch("paper.orchestrator.Path.unlink")
    def test_teardown_removes_config(self, mock_unlink, mock_subprocess):
        """teardown_paper_instance deletes config file."""
        instance = PaperInstance(
            strategy_name="TestStrat",
            port=8090,
            container_name="paper_teststrat_0",
            db_schema="paper_teststrat",
        )

        teardown_paper_instance(instance)

        # Should call unlink on config file
        mock_unlink.assert_called()


class TestTeardownAll:
    """Test bulk teardown of all paper instances."""

    @patch("paper.orchestrator.subprocess.run")
    @patch("paper.orchestrator.Path.exists")
    @patch("paper.orchestrator.Path.glob")
    @patch("paper.orchestrator.Path.unlink")
    def test_teardown_all_lists_containers(self, mock_unlink, mock_glob, mock_exists, mock_subprocess):
        """teardown_all queries podman for all paper_ containers."""
        mock_exists.return_value = True
        mock_glob.return_value = []
        mock_subprocess.return_value = MagicMock(
            returncode=0, stdout="paper_strat1_0\npaper_strat2_1\n"
        )

        teardown_all()

        # Should call podman ps -a with paper_ filter
        first_call = mock_subprocess.call_args_list[0][0][0]
        assert "podman" in first_call
        assert "ps" in first_call
        # Check that filter includes 'paper_'
        call_str = " ".join(first_call)
        assert "paper_" in call_str


class TestListPaperInstances:
    """Test listing running paper instances."""

    @patch("paper.orchestrator.subprocess.run")
    def test_list_instances(self, mock_subprocess):
        """list_paper_instances returns container names."""
        mock_subprocess.return_value = MagicMock(
            stdout="paper_strat1_0\npaper_strat2_1\n"
        )

        instances = list_paper_instances()

        assert len(instances) == 2
        assert "paper_strat1_0" in instances
        assert "paper_strat2_1" in instances

    @patch("paper.orchestrator.subprocess.run")
    def test_list_instances_empty(self, mock_subprocess):
        """list_paper_instances returns empty list if no containers."""
        mock_subprocess.return_value = MagicMock(stdout="")

        instances = list_paper_instances()

        assert instances == []

    @patch("paper.orchestrator.subprocess.run")
    def test_list_instances_filters_empty_lines(self, mock_subprocess):
        """list_paper_instances filters out empty lines."""
        mock_subprocess.return_value = MagicMock(
            stdout="paper_strat1_0\n\npaper_strat2_1\n\n"
        )

        instances = list_paper_instances()

        assert len(instances) == 2
        assert "" not in instances
