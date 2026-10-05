# Runbook — Hyperliquid Agent Trading Stack

How the stack is deployed and run. Architecture and reasoning: [design.md](design.md).
Current state: [STATUS.md](STATUS.md).

Everything about this stack lives in this repo: code, Ansible, and docs. Deploys
run from the Mac (`make deploy`) because the playbooks read secrets from gopass.

## Services

| Host | Service (user systemd) | Port | What it does |
|---|---|---|---|
| tela | `backtest-api.service` | 127.0.0.1:8070 | `POST /backtest` — Phase 1 numpy filter, Phase 2 Freqtrade walk-forward |
| trinity | `trading-postgres.service` | 127.0.0.1:5432 | `strategy_registry`, `paper_snapshots` |
| trinity | `paper-arena-monitor.service` | host network | Runs queued strategies as Freqtrade dry-run containers for 14 days, snapshots hourly |
| trinity | `paper_<name>_<slot>` containers | 127.0.0.1:8090–8095 | One Freqtrade dry-run per paper candidate, started by the monitor |
| trinity | `trading-live.service` | 127.0.0.1:8080 | The live Freqtrade bot. **Dry-run by default**, strategy `NullStrategy` until promoted |

## Web UIs

On the `eddiedunn.github` tailnet (tela and trinity's tailnet), through Caddy
with the `*.starbluesolutions.net` wildcard cert. The routes live in
starblue-infra, which owns Caddy; the names resolve through pfSense host
overrides to each box's Tailscale address.

| Address | What |
|---|---|
| https://trading.starbluesolutions.net | Live bot FreqUI (trinity 8080) |
| https://paper.starbluesolutions.net | Paper arena slot 0 FreqUI (trinity 8090) |
| https://backtest.starbluesolutions.net/docs | Backtest API page (tela 8070) |

FreqUI login: user `freqtrade`, password `gopass show trading/freqtrade-api-password`.
Open each bot in its own tab; one FreqUI showing both bots would need
`CORS_origins` set in the bot configs. Paper slots 1–5 (8091–8095) have no
names yet.

Without the tailnet: `ssh -N -L 8080:127.0.0.1:8080 -L 8090:127.0.0.1:8090 trinity`
and open http://localhost:8080 / :8090. The scripts reach the backtest API
through `ssh tela`.

### Data locations

| Path | Host | Contents |
|---|---|---|
| `/data/services/backtest_api/data` | tela | 4h candles + hourly funding, Phase 1 format |
| `/data/services/backtest_api/ftdata` | tela | The same, exported in Freqtrade's futures layout |
| `/data/services/backtest_api/{strategies,results}` | tela | Posted candidates, Freqtrade results per strategy/window |
| `/data/services/trading_postgres/data` | trinity | Postgres data dir |
| `/data/services/trading/paper` | trinity | Paper candidates, per-slot configs, logs, dry-run trade DBs |
| `/opt/trading` | trinity | Live bot: `live/config.json`, `live/active_strategy.txt`, trade DB |

Each data directory is mounted at the **same path** inside its container. The
backtest API and the paper monitor start sibling Freqtrade containers through
the rootless podman socket, and same-path mounts make the paths they pass
resolve on the host.

## Deploy

```bash
make test            # 127 unit tests
make deploy          # postgres → backtest API → paper arena → live bot (dry-run)
make smoketest       # health, units, tables, live bot dry-run, Phase 1 round-trip
```

Single pieces: `make deploy-backtest`, `deploy-postgres`, `deploy-paper`,
`deploy-live`. Pass `-e net_path=mesh` inside `deploy/ansible` to go over
Tailscale instead of the home LAN.

Each backtest deploy also tops up price and funding history from Hyperliquid
(about 2 s per request to stay under its rate limit; a few seconds when already
current). Hyperliquid only serves the last 5000 candles, so the history from
before that (back to 2023-12) exists only in the files already on tela and in
this repo's local `data/` folder, which the role seeds from when tela has none.
There is no scheduled download; re-deploy, or run the download task's command,
to refresh.

Secrets (gopass): `trading/postgres-password`, `trading/freqtrade-api-password`,
and — only read for real-money mode — `trading/hyperliquid-private-key`,
`trading/hyperliquid-wallet-address`.

## Strategy file format

One file per strategy, named after its class. It serves every phase:

- a module-level `generate_signals(df) -> Series` of 0/1 positions for Phase 1, and
- a Freqtrade `IStrategy` subclass for Phases 2–4.

See `strategies/examples/EmaCross.py`. Freqtrade is optional at import so Phase 1
can load the file without it.

## Run a strategy through the pipeline

```bash
scripts/submit_strategy.sh path/to/MyStrat.py          # Phase 1 + 2, queue for paper if both pass
scripts/submit_strategy.sh path/to/MyStrat.py --force  # queue anyway (pipeline testing only)
```

The monitor checks the queue every 5 minutes and runs everything queued (up to
6) as one cohort for 14 days. A monitor restart re-queues the running cohort and
its 14 days start over. Outcomes land in `strategy_registry.paper_passed`.

Watch it:

```bash
ssh trinity podman logs -f paper-arena-monitor
ssh trinity podman exec trading-postgres psql -U trading -d trading \
  -c "select ts, strategy, profit_pct, trade_count, win_rate, profit_factor, max_drawdown from paper_snapshots order by ts desc limit 20"
```

## Strategy agent

`python -m agent run` asks Claude (through `claude -p`, so it uses the
Claude subscription login, not an API key) for a strategy in the two-in-one format,
checks it locally (parses, has `generate_signals` and the right class, imports
only pandas/numpy/pandas_ta/freqtrade, at most 6 constants, no negative
`shift`), posts it to Phase 1, and on failure sends the metrics back for a
revision. Phase 2 runs only after Phase 1 passes; Claude then sees in-sample
and validation metrics but only pass/fail for the out-of-sample window.

```bash
ssh -N -L 8070:127.0.0.1:8070 tela &                 # the API only listens on tela's loopback
uv run python -m agent run --max-strategies 2 --seed "Donchian breakout with a volume filter"
```

Caps: `--max-iterations 5` Claude revisions per strategy, `--max-phase2 2`
Phase 2 runs per strategy (minutes each), `--max-strategies 2` per run. Model:
`--model` (default `claude-opus-5-5`). API: `--api-url` or `BACKTEST_API_URL`.

Every attempt lands in `agent_runs/<run id>/log.jsonl` (gitignored) with each
file version next to it. A strategy that passes both phases is saved as
`agent_runs/<run id>/<Name>.py` with its `phase1.json` / `phase2.json`, and the
agent prints the `scripts/submit_strategy.sh` and `paper-add` commands for you
to run. Only `--queue-paper` makes it call `paper-add` on trinity itself, never
with `--force`. The agent has no code path to `promote`.

## Promote to the live bot (manual)

1. Review `paper_snapshots` and the strategy code.
2. `ssh trinity` and, with `POSTGRES_*` set, `python -m live.trading_client promote --strategy <Name>`
   (copies the file into `/opt/trading/live`, writes `active_strategy.txt`, marks the registry).
3. `systemctl --user restart trading-live.service`.
4. `trading_client retire --strategy <Name>` puts `NullStrategy` back on the next restart.

The live bot is still dry-run after a promotion. Real money is a separate,
deliberate step:

```bash
cd deploy/ansible && ansible-playbook playbooks/deploy_trading_live.yml -e trading_live_dry_run=false
# asserts the wallet secrets exist, then asks you to type LIVE
```

## Testnet (`trading_env`)

All trinity roles accept `-e trading_env=testnet`: gopass paths move from
`trading/*` to `trading-test/*`, and Freqtrade's exchange URL points at
`api.hyperliquid-testnet.xyz` for the live bot and every paper instance. The
`trading-test/*` secrets do not exist yet. Testnet and prod share container
names and ports, so only one can run at a time.

## Known gotchas

- Freqtrade cannot download Hyperliquid history (`download-data` refuses), which
  is why `scripts/download_data.py --freqtrade-dir` writes Freqtrade's files.
  Its futures backtester also needs hourly mark and funding files; mark candles
  are the 4h candles forward-filled to 1h, since Hyperliquid keeps only ~7
  months of 1h candles.
- Old Freqtrade images (Feb 2026) crash loading Hyperliquid's market list
  (`TypeError: ... 'NoneType' and 'str'` in ccxt `fetch_spot_markets`). Every
  role re-pulls `freqtradeorg/freqtrade:stable`.
- Sibling Freqtrade containers run with `--userns=keep-id:uid=1000,gid=1000`,
  so their output files belong to the host user and the calling container can
  read them.
- Phase 1 has no stoploss or trailing stop; Phase 2 uses `config/backtest.json`
  (5% stop, 2% trailing). Expect the two to disagree on the same strategy.
