# Hyperliquid Agent Trading Stack

Autonomous agent-driven perpetual futures trading on Hyperliquid. Four-phase pipeline: numpy fast filter, Freqtrade walk-forward, paper arena, live bot.

Code, deployment (Ansible in `deploy/ansible/`) and docs all live here:

- [docs/runbook.md](docs/runbook.md) — services, deploy, running a strategy through the pipeline
- [docs/design.md](docs/design.md) — architecture and reasoning
- [docs/STATUS.md](docs/STATUS.md) — what works today

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

From the Mac (the playbooks read secrets from gopass):

```bash
make deploy      # tela: backtest API; trinity: postgres, paper arena, live bot (dry-run)
make smoketest
scripts/submit_strategy.sh strategies/examples/EmaCross.py --force
```

See [docs/runbook.md](docs/runbook.md).

## Structure

```
backtest_api/     Phase 1 numpy fast filter + Phase 2 Freqtrade walk-forward
scripts/          OHLCV + funding download, smoketest, submit_strategy
paper/            Paper arena orchestrator + monitor
live/             Live promotion CLI (trading_client)
sql/              Postgres schema
strategies/       NullStrategy, examples/ (two-in-one strategy file format)
config/           Freqtrade config templates
tests/            Unit tests
deploy/           Dockerfiles + Ansible (inventory, roles, playbooks)
docs/             Runbook, design, status
```

## Live Promotion Workflow

Promotion from paper to live is **manual** — `paper/monitor.py` only logs a candidate; an operator must review and run the CLI.

1. `paper/monitor.py` finishes its 14-day window and logs `PROMOTION CANDIDATE: <name> — run ...`.
2. Operator inspects `paper_snapshots` and the candidate strategy code.
3. Promote: `python -m live.trading_client promote --strategy <name>` — copies the strategy into the live slot, writes `/opt/trading/live/active_strategy.txt`, and sets `promoted_live=true` / `promoted_at=now()` in `strategy_registry`. Restart the live bot to pick up the new strategy.
4. `python -m live.trading_client status` — show currently promoted strategy + timestamp.
5. `python -m live.trading_client retire --strategy <name>` — mark retired in registry; live reverts to `NullStrategy` on next restart.

Requires `POSTGRES_HOST`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` in env.
