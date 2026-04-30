# Hyperliquid Agent Trading Stack

Autonomous agent-driven perpetual futures trading on Hyperliquid. Four-phase pipeline: numpy fast filter, Freqtrade walk-forward, paper arena, live bot.

**Design doc:** See `starblue-infra/docs/services/trading-stack.md` for full architecture.

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

All deployment is owned by **starblue-infra**. From that repo:

```bash
ansible-playbook -i inventory/vps/hosts.yml playbooks/deploy_backtest_api.yml         # tela
ansible-playbook -i inventory/vps/hosts.yml playbooks/deploy_trading_postgres.yml     # trinity
ansible-playbook -i inventory/vps/hosts.yml playbooks/deploy_trading_paper_arena.yml  # trinity
ansible-playbook -i inventory/vps/hosts.yml playbooks/deploy_trading_live.yml         # trinity
```

- **tela:** `backtest_api` service (FastAPI on `127.0.0.1:8070`).
- **trinity:** `trading_postgres` (5432), `trading_paper_arena` (monitor + dynamic 8090–8095), `trading_live` (Freqtrade on 8080).

Secrets (gopass): `trading/postgres-password`, `trading/freqtrade-api-password`, `trading/hyperliquid-private-key`, `trading/hyperliquid-wallet-address`.

## Structure

```
backtest_api/     Phase 1 numpy fast filter + Phase 2 Freqtrade walk-forward
scripts/          OHLCV data collection (daily cron)
paper/            Paper arena orchestrator + monitor
live/             Live promotion CLI (trading_client)
sql/              Postgres schema
strategies/       Agent-written strategies (candidates/)
config/           Freqtrade config templates
tests/            Unit tests
```

## Live Promotion Workflow

Promotion from paper to live is **manual** — `paper/monitor.py` only logs a candidate; an operator must review and run the CLI.

1. `paper/monitor.py` finishes its 14-day window and logs `PROMOTION CANDIDATE: <name> — run ...`.
2. Operator inspects `paper_snapshots` and the candidate strategy code.
3. Promote: `python -m live.trading_client promote --strategy <name>` — copies the strategy into the live slot, writes `/opt/trading/live/active_strategy.txt`, and sets `promoted_live=true` / `promoted_at=now()` in `strategy_registry`. Restart the live bot to pick up the new strategy.
4. `python -m live.trading_client status` — show currently promoted strategy + timestamp.
5. `python -m live.trading_client retire --strategy <name>` — mark retired in registry; live reverts to `NullStrategy` on next restart.

Requires `POSTGRES_HOST`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` in env.
