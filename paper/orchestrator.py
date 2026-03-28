"""Paper arena orchestrator — spawn/teardown Freqtrade dry_run containers.

Each candidate strategy gets its own Freqtrade container with a unique port
and isolated config. Port range: 8090-8095 (6 slots max on trinity).
"""

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

BASE_PORT = 8090  # 8090, 8091, 8092 ... per candidate
MAX_SLOTS = 6

PAPER_CONFIGS_DIR = _REPO_ROOT / "paper" / "configs"


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
    PAPER_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    config_path = PAPER_CONFIGS_DIR / f"{container_name}.json"
    config_path.write_text(json.dumps(config, indent=2))

    strategies_dir = str(_REPO_ROOT / "strategies")
    data_dir = str(_REPO_ROOT / "data")
    logs_dir = str(_REPO_ROOT / "logs")

    subprocess.run(
        [
            "podman", "run", "-d",
            "--name", container_name,
            "-v", f"{strategies_dir}:/freqtrade/strategies:ro,Z",
            "-v", f"{data_dir}:/freqtrade/user_data/data:ro,Z",
            "-v", f"{str(PAPER_CONFIGS_DIR)}:/freqtrade/config:ro,Z",
            "-v", f"{logs_dir}:/freqtrade/logs:Z",
            "-p", f"127.0.0.1:{port}:{port}",
            "freqtradeorg/freqtrade:stable",
            "trade",
            "--config", f"/freqtrade/config/{container_name}.json",
            "--strategy", strategy_name,
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
    config_path = PAPER_CONFIGS_DIR / f"{instance.container_name}.json"
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
    if PAPER_CONFIGS_DIR.exists():
        for f in PAPER_CONFIGS_DIR.glob("paper_*.json"):
            f.unlink(missing_ok=True)


def list_paper_instances() -> list[str]:
    """List running paper container names."""
    result = subprocess.run(
        ["podman", "ps", "--filter", "name=paper_", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    return [n for n in result.stdout.strip().splitlines() if n]


def _build_paper_config(strategy_name: str, port: int, db_schema: str) -> dict:
    """Build Freqtrade config for a paper instance."""
    return {
        "exchange": {
            "name": "hyperliquid",
            "options": {"defaultType": "swap"},
        },
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "stake_currency": "USDC",
        "stake_amount": 33,
        "max_open_trades": 3,
        "timeframe": "4h",
        "pair_whitelist": ["BTC/USDC:USDC", "ETH/USDC:USDC", "SOL/USDC:USDC"],
        "dry_run": True,
        "dry_run_wallet": 100,
        "stoploss": -0.05,
        "trailing_stop": True,
        "trailing_stop_positive": 0.02,
        "api_server": {
            "enabled": True,
            "listen_ip_address": "0.0.0.0",
            "listen_port": port,
            "username": "freqtrade",
            "password": "changeme",
            "jwt_secret_key": "generate-a-real-secret-here",
        },
    }
