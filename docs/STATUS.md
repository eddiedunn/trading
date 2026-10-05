# Project Status — Hyperliquid Agent Trading Stack

_As of 2026-10-02._

| Phase | Host | Status |
|---|---|---|
| 1. Fast filter | tela | **Deployed, working** — `POST /backtest` phase 1 |
| 2. Walk-forward | tela | **Deployed, working** — 3 rolling windows ending today (12/6/6 months), real Freqtrade runs |
| 3. Paper arena | trinity | **Deployed, working** — queue via `scripts/submit_strategy.sh` (Phase 1, Phase 2, final test), runs until 30 trades or 60 days, must match a backtest of the same period |
| 4. Live bot | trinity | **Deployed in dry-run** with `NullStrategy`; real money needs `-e trading_live_dry_run=false` + typed confirmation |
| Agent loop | — | **Not built** — nothing writes strategies on its own yet (design.md rollout step 6) |

First end-to-end run (2026-10-02): `strategies/examples/EmaCross.py` failed
Phase 1 and all three Phase 2 windows (profit factor 0.11–0.24) and was queued
with `--force` to exercise the paper arena.

## Next steps

1. The agent loop: something that writes candidates and calls `submit_strategy.sh`.
2. A scheduled data refresh (today: re-deploy the backtest API).
3. Testnet secrets (`trading-test/*`) if a testnet stage is wanted before real money.
4. Public DNS for `backtest.starbluesolutions.net` did not answer from the Mac on 2026-10-02.
