# Hyperliquid Agent Trading Stack

Autonomous agent-driven perpetual futures trading on Hyperliquid. Four-phase pipeline: numpy fast filter, Freqtrade walk-forward, paper arena, live bot.

**Design doc:** See `claw-deploy/docs/hyperliquid-agent-trading-stack.md` for full architecture.

## Quick Start

```bash
# Install dependencies
uv sync

# Download OHLCV data
uv run python scripts/download_data.py

# Run backtest API (tela)
uv run uvicorn backtest_api.main:app --host 127.0.0.1 --port 8070

# Run tests
uv run pytest
```

## Deployment

This repo is cloned to both hosts by Ansible:
- **tela:** backtest API service (starblue-infra `roles/backtest_api/`)
- **trinity:** paper arena, live bot, Postgres (claw-deploy `roles/trading/`)

## Structure

```
backtest_api/     Phase 1 numpy fast filter + Phase 2 Freqtrade walk-forward
scripts/          OHLCV data collection (daily cron)
paper/            Paper arena orchestrator + monitor
sql/              Postgres schema
strategies/       Agent-written strategies (candidates/)
config/           Freqtrade config templates
tests/            Unit tests
```
