# Project Status — Hyperliquid Agent Trading Stack

_As of 2026-04-30, branch `main` clean._

## Pipeline Overview

Four-phase strategy lifecycle:

| Phase | Purpose | Host | Status |
|-------|---------|------|--------|
| 1. Fast filter | numpy sub-ms screen of signal variants | tela | **Implemented + tested** |
| 2. Walk-forward | Freqtrade IS / validation / OOS windows | tela | **Implemented**, no tests |
| 3. Paper arena | 14-day dry-run on up to 6 candidates | trinity | **Implemented**, no tests |
| 4. Live bot | Production trading of promoted strategy | trinity | **Stub only** (`NullStrategy`) |

## Module Inventory

### `backtest_api/` — Phases 1 & 2 (FastAPI on tela:8070)
- **`main.py`** — `POST /backtest` endpoint. Pydantic `BacktestRequest` (strategy_name, strategy_code, phase, timerange). Writes candidate to `strategies/candidates/`, dispatches to filter/walker.
- **`fast_filter.py`** — Phase 1 core. `run_fast_filter()` dynamically imports strategy, calls `generate_signals(df)`, computes 7 metrics (return, sharpe, max_dd, win_rate, profit_factor, trade_count, calmar). Applies TAKER_FEE 0.045% and synthetic FUNDING_RATE_PER_BAR 0.01%. Gate `meets_phase1_criteria()`: return>20%, DD>-20%, PF>1.3, WR>45%, trades>30, sharpe>0.8.
- **`walk_forward.py`** — Phase 2. Spawns `freqtradeorg/freqtrade:stable` podman container against three windows (2023-01→2024-06 IS, 2024-06→2025-06 val, 2025-06→2026-01 OOS). Per-window gate: PF≥1.2, DD≥-25%.

### `paper/` — Phase 3 (trinity)
- **`orchestrator.py`** — `spawn_paper_instance(strategy, slot)` launches Freqtrade dry-run on port 8090+slot. Generates per-candidate JSON config: `dry_run_wallet=100 USDC`, `stake_amount=33`, `max_open_trades=3`, `timeframe=4h`, pairs BTC/ETH/SOL.
- **`monitor.py`** — `run_paper_arena()` polls Freqtrade REST hourly for 14 days, persists `paper_snapshots` to Postgres, selects best performer. Gate `meets_promotion_criteria()`: trades≥20, profit≥5%, DD≥-15%, WR≥45%, PF≥1.25.

### `scripts/`
- **`download_data.py`** — ccxt-based Hyperliquid OHLCV downloader. Idempotent, paginates 5000-candle chunks, dedupes, writes feather. CLI: `--pairs`, `--timeframes`, `--data-dir`, `--since`. Cron-safe.

### `strategies/`
- **`NullStrategy.py`** — Freqtrade INTERFACE_VERSION=3 placeholder; never enters positions. Live bot's default until promotion path exists.
- **`candidates/`** — empty; agent-written strategies land here on `/backtest`.

### `config/`
- **`backtest.json`** — Freqtrade Phase 2 template (hyperliquid futures, USDC, 4h, dry_run, -5% SL, trailing).
- `paper_template.json` / `live_template.json` — present but unused; orchestrator builds configs dynamically.

### `sql/init.sql`
- `paper_snapshots(strategy, ts, ...)` — hourly metrics, indexed on (strategy, ts).
- `strategy_registry(name, parent_strategy, iteration, phase1/2/paper passed, promoted_live, ...)` — full lifecycle tracking.

### `tests/`
- **`test_fast_filter.py`** — 247 lines, full coverage of Phase 1: `_compute_metrics`, `_aggregate_metrics`, `meets_phase1_criteria`, integration with synthetic OHLCV.

### `deploy/`
- **`Dockerfile`** — Python 3.13-slim + uv, exposes 8070, healthcheck `/health`.
- **`backtest-api.container`** — Podman Quadlet systemd unit for tela.
- **`ansible-deploy.yml`** — syncs repo, copies `data/` + `config/` to `/opt/trading`, reloads Quadlet.

### `data/`
9 feather files: BTC/ETH/SOL × {1h, 4h, 1d}, ~1.2 MB.

## CI/CD (`Jenkinsfile`)

4 stages on Linux GPU agent:
1. **Trivy scan** (vuln + secret, fail HIGH/CRITICAL)
2. **Tests** — `uv sync --extra dev --frozen` → `pytest --cov=backtest_api --cov-report=xml`
3. **SonarQube** — scans `backtest_api/`, `paper/`, `scripts/`; uploads coverage
4. **Deploy** — `ansible-playbook deploy/ansible-deploy.yml`

Build images now route through the Starblue Zot registry.

## Gaps & Known TODOs

1. **Phase 4 live promotion** — no `trading_client.py`, no mechanism to swap the live bot from `NullStrategy` to a paper-arena winner.
2. **Test coverage** — only `fast_filter.py` is tested; `walk_forward`, `orchestrator`, `monitor` have no unit tests.
3. **Secrets** — Freqtrade REST auth hardcoded `freqtrade/changeme` in `paper/monitor.py`; needs env-var/secret-store.
4. **Unused config templates** — `config/paper_template.json` and `config/live_template.json` exist but orchestrator generates configs dynamically; either wire them in or delete.
5. **Log collection** — `logs/` is empty; no aggregator pulls Freqtrade container logs.

## Recent Activity

```
d4280d2 chore(ci): route build images through zot
608b34f feat: add NullStrategy placeholder for live bot initial start
5a10716 fix: add pytest-cov for CI coverage + fix deploy permissions
9ff05f8 fix: commit uv.lock for reproducible CI builds
bbd5a51 feat: full repo scaffolding + Jenkins CI/CD pipeline
```
