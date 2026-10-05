"""Paper arena orchestrator — spawn/teardown Freqtrade dry_run containers.

Each candidate strategy gets its own Freqtrade container with a unique port
and isolated config. Port range: 8090-8095 (6 slots max on trinity).
"""

import json
import os
import secrets
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

FREQTRADE_IMAGE = "docker.io/freqtradeorg/freqtrade:stable"

BASE_PORT = 8090  # 8090, 8091, 8092 ... per candidate
MAX_SLOTS = 6


def paper_dir() -> Path:
    """Root for candidate strategies, per-slot configs and logs.

    When the monitor runs in a container, this directory is bind-mounted at the
    same path it has on the host, so the paths passed to sibling Freqtrade
    containers (spawned through the podman socket) resolve on the host.
    """
    return Path(os.environ.get("TRADING_PAPER_DIR", str(_REPO_ROOT / "paper" / "run")))


def strategies_dir() -> Path:
    return paper_dir() / "strategies"


def configs_dir() -> Path:
    return paper_dir() / "configs"


@dataclass
class PaperInstance:
    strategy_name: str
    port: int
    container_name: str
    db_schema: str

    def to_dict(self) -> dict:
        return asdict(self)


def spawn_paper_instance(strategy_name: str, slot: int) -> PaperInstance:
    """Start a paper trading container for a candidate strategy."""
    if slot >= MAX_SLOTS:
        raise ValueError(f"Slot {slot} exceeds max {MAX_SLOTS} (ports {BASE_PORT}-{BASE_PORT + MAX_SLOTS - 1})")

    port = BASE_PORT + slot
    container_name = f"paper_{strategy_name.lower()}_{slot}"
    db_schema = f"paper_{strategy_name.lower()}"

    # Write per-candidate config
    config = _build_paper_config(strategy_name, port, db_schema)
    configs_dir().mkdir(parents=True, exist_ok=True)
    config_path = configs_dir() / f"{container_name}.json"
    config_path.write_text(json.dumps(config, indent=2))

    logs_dir = paper_dir() / "logs" / strategy_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "podman", "run", "-d",
            "--name", container_name,
            # Run as the host user so logs and the trade DB stay host-owned.
            "--userns=keep-id:uid=1000,gid=1000",
            "-v", f"{strategies_dir()}:/freqtrade/strategies:ro,Z",
            "-v", f"{configs_dir()}:/freqtrade/config:ro,Z",
            "-v", f"{logs_dir}:/freqtrade/logs:Z",
            "-p", f"127.0.0.1:{port}:{port}",
            FREQTRADE_IMAGE,
            "trade",
            "--config", f"/freqtrade/config/{container_name}.json",
            "--strategy", strategy_name,
            "--strategy-path", "/freqtrade/strategies",
            "--logfile", "/freqtrade/logs/freqtrade.log",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return PaperInstance(strategy_name, port, container_name, db_schema)


def teardown_paper_instance(instance: PaperInstance):
    """Stop and remove a paper trading container."""
    subprocess.run(["podman", "stop", instance.container_name], check=False, capture_output=True)
    subprocess.run(["podman", "rm", instance.container_name], check=False, capture_output=True)
    config_path = configs_dir() / f"{instance.container_name}.json"
    config_path.unlink(missing_ok=True)


def teardown_all():
    """Stop and remove all paper trading containers."""
    result = subprocess.run(
        ["podman", "ps", "-a", "--filter", "name=paper_", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    for name in result.stdout.strip().splitlines():
        if name:
            subprocess.run(["podman", "stop", name], check=False, capture_output=True)
            subprocess.run(["podman", "rm", name], check=False, capture_output=True)

    # Clean up config files
    if configs_dir().exists():
        for f in configs_dir().glob("paper_*.json"):
            f.unlink(missing_ok=True)


def list_paper_instances() -> list[str]:
    """List running paper container names."""
    result = subprocess.run(
        ["podman", "ps", "--filter", "name=paper_", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    return [n for n in result.stdout.strip().splitlines() if n]


_TESTNET_API_URL = "https://api.hyperliquid-testnet.xyz"

PAIRS = ["BTC/USDC:USDC", "ETH/USDC:USDC", "SOL/USDC:USDC"]


def _build_paper_config(strategy_name: str, port: int, db_schema: str) -> dict:
    """Build Freqtrade config for a paper instance."""
    exchange: dict = {
        "name": "hyperliquid",
        "ccxt_config": {"options": {"defaultType": "swap"}},
        "pair_whitelist": PAIRS,
    }
    if os.environ.get("TRADING_ENV") == "testnet":
        exchange["ccxt_config"]["urls"] = {
            "api": {"public": _TESTNET_API_URL, "private": _TESTNET_API_URL}
        }
    return {
        "exchange": exchange,
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "stake_currency": "USDC",
        "stake_amount": 33,
        "max_open_trades": 3,
        "timeframe": "4h",
        "dry_run": True,
        "dry_run_wallet": 100,
        "pairlists": [{"method": "StaticPairList"}],
        "entry_pricing": {"price_side": "same"},
        "exit_pricing": {"price_side": "same"},
        # Keep the dry-run trade history next to the logs so it survives restarts.
        "db_url": "sqlite:////freqtrade/logs/tradesv3.dryrun.sqlite",
        "initial_state": "running",
        # No stoploss, trailing_* or minimal_roi: config values override the
        # strategy's own, and each strategy declares its exits.
        "api_server": {
            "enabled": True,
            "listen_ip_address": "0.0.0.0",
            "listen_port": port,
            "username": os.environ.get("FREQTRADE_API_USER", "freqtrade"),
            "password": os.environ["FREQTRADE_API_PASSWORD"],
            "jwt_secret_key": secrets.token_hex(32),
        },
    }
