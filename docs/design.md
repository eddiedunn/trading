> Moved here from `claw-deploy/docs/` on 2026-10-02. This is the original March 2026 design; for how it is deployed today see [runbook.md](runbook.md).

# Hyperliquid Agent Trading Stack
**Design Document — March 28, 2026**

---

## Overview

A fully autonomous, agent-driven trading system targeting Hyperliquid perpetuals. No TradingView subscription, no Signum middleware, no SaaS dependencies. Agents write strategies, backtest them, validate them, and control a live trading bot — all on self-hosted infrastructure.

**Goals:**
- Cheap and reliable over whiz-bang UI
- Agent-native: programmatic at every layer
- No KYC exchange (Hyperliquid, wallet-based auth)
- Self-hosted across existing infrastructure (trinity + tela)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        TRINITY (orchestrator)                        │
│                   OpenClaw agent — always-on                         │
│                                                                       │
│  - writes/modifies Python strategies                                 │
│  - calls tela backtest API, reads results                            │
│  - decides to promote or iterate                                     │
│  - runs paper arena, monitors live bot                               │
│  - emergency stop / reconfig                                         │
│  - Postgres: strategy registry, paper snapshots, trade history       │
└──────────┬───────────────────────────────────────┬───────────────────┘
           │  POST /backtest (HTTPS)               │  local
           ▼                                       ▼
┌──────────────────────────┐      ┌────────────────────────────────────┐
│   TELA (compute)         │      │  TRINITY (long-running services)   │
│   i5-13600K 14c 94GB     │      │  4-core i3, 32GB                   │
│                          │      │                                    │
│  PHASE 1   │  PHASE 2    │      │  PHASE 2.5        │  PHASE 3       │
│  numpy     │  Freqtrade  │      │  Paper Arena      │  Live Bot      │
│  fast iter │  backtest   │      │  N×Freqtrade      │  Freqtrade     │
│  <1ms/run  │  walk-fwd   │      │  dry_run: true    │  dry_run: false│
│  14c burst │  3 windows  │      │  4-6 containers   │  single winner │
│  own OHLCV │  own OHLCV  │      │  lightweight      │  REST API      │
└────────────┴─────────────┘      └───────────────────┴────────────────┘
                                                               │
                                                               │ ccxt.hyperliquid
                                                               ▼
                                                    ┌─────────────────────┐
                                                    │    HYPERLIQUID      │
                                                    │  378 perp markets   │
                                                    │  USDC settled       │
                                                    │  No KYC             │
                                                    │  Wallet-based auth  │
                                                    └─────────────────────┘
```

---

## Why This Stack

### Why NOT TradingView + Signum
- TradingView is optimized for humans looking at charts — agents don't need the UI
- PineScript runs in a sandboxed VM you can't programmatically access
- Signum is middleware that adds cost and a failure point
- Combined cost: ~$60-100/mo for features agents don't use

### Why Freqtrade
- Open source, battle-tested, active community
- Built-in backtesting with realistic fee/slippage modeling
- Same Python code runs backtests AND live — no translation layer
- Full REST API for agent control
- ccxt handles exchange connectivity (100+ exchanges, same interface)
- Runs in a single Docker/Podman container

### Why Custom numpy/pandas for Phase 1
- Vectorized — sub-millisecond per evaluation, 10,000 variants in under 10 seconds
- Zero framework dependencies — just pandas + numpy
- Agent writes a plain function (signal logic), helper module computes metrics
- Simplest possible API surface for agent-generated code — no classes to subclass
- Filters out bad strategies before expensive Freqtrade validation
- VectorBT was considered but rejected: OSS version frozen since 2022, no futures/funding modeling, Numba cold-start overhead unnecessary for this use case

### Why Hyperliquid
- #1 DEX by derivatives volume
- No KYC — wallet private key auth
- Deep liquidity, low fees
- Supports: longs, shorts, leverage, isolated margin
- 378 perp markets, 284 spot markets
- Fully supported by ccxt (confirmed working)
- Freqtrade has Hyperliquid in its constants (confirmed)

### Why Tela for Backtesting, Trinity for Live

Backtesting is bursty CPU + RAM work — run hard for minutes, then done. Tela's i5-13600K (14 cores, 94GB RAM) vs trinity's 4-core i3 is the obvious fit. Phase 1 numpy sweeps parallelize trivially across cores via multiprocessing.

Live and paper trading are the opposite: long-running, mostly sleeping, woken by candle closes on a 4h timeframe. Trinity handles this trivially and keeps all persistent state (Postgres, logs, backups) in one place alongside OpenClaw.

Tela's GPU (RTX 4090) is not used by the trading stack — it's already occupied by sentinel v2, embed.api, and diarized_transcriber. Backtesting is CPU/RAM only.

OHLCV data is duplicated by design: tela maintains its own local feather copy (downloaded direct from Hyperliquid) so backtests don't depend on a trinity network mount. Trinity holds the canonical copy for the paper and live bots.

### PineScript Portability
- Popular indicators already exist in `pandas-ta` (130+ indicators)
- Gaussian Channel, Supertrend, Ichimoku, MACD, Bollinger — all available
- Claude can translate any remaining PineScript → Python in one shot
- Key translation notes:
  - Pine `close[1]` → Python `df['close'].shift(1)`
  - Pine `security()` → multi-timeframe handling in pandas
  - Repainting bugs actually get *fixed* in translation if using confirmed bars

---

## Infrastructure

### Host Responsibilities

| Concern | Trinity | Tela |
|---|---|---|
| Hardware | 4-core i3-8109U, 32GB, 3.7TB NVMe | i5-13600K 14-core, 94GB RAM, RTX 4090 |
| Phase 1 numpy fast filter | — | ✓ (CPU burst, 14 cores) |
| Phase 2 Freqtrade backtest | — | ✓ (CPU burst, 14 cores) |
| Phase 2.5 paper arena | ✓ (long-running, lightweight) | — |
| Phase 3 live bot | ✓ | — |
| Postgres | ✓ Podman container (B2 backups already wired) | — |
| OHLCV feather files | ✓ canonical copy | ✓ local copy for backtesting |
| OpenClaw agent | ✓ | — |
| Backtest API service | — | ✓ (FastAPI + Caddy HTTPS) |

### Tela — Backtest API Service

A lightweight FastAPI service accepts strategy code from the agent on trinity, runs the requested phase, and returns metrics. Caddy handles TLS the same way as sentinel and embed.api.

**URL:** `https://backtest.starbluesolutions.net`

```yaml
# tela: trading/podman-compose.yml
version: "3"

services:
  backtest-api:
    image: python:3.12-slim
    restart: unless-stopped
    volumes:
      - ./backtest_api:/app
      - ./strategies:/strategies
      - ./data:/freqtrade/user_data/data
      - ./config:/freqtrade/config:ro
    ports:
      - "127.0.0.1:8070:8070"
    working_dir: /app
    command: >
      uv run uvicorn main:app --host 0.0.0.0 --port 8070

  freqtrade-backtest:
    image: freqtradeorg/freqtrade:stable
    volumes:
      - ./strategies:/freqtrade/strategies
      - ./data:/freqtrade/user_data/data
      - ./config:/freqtrade/config:ro
      - ./backtest_results:/freqtrade/user_data/backtest_results
    profiles: ["backtest"]   # only started on demand by API, not persistent
```

```python
# tela: trading/backtest_api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import subprocess, json
from fast_filter import run_fast_filter, meets_phase1_criteria
from walk_forward import walk_forward_test

app = FastAPI()

class BacktestRequest(BaseModel):
    strategy_name: str
    strategy_code: str
    phase: int          # 1 = VectorBT fast filter, 2 = Freqtrade walk-forward
    timerange: str = "20230101-20260101"

@app.post("/backtest")
def run_backtest(req: BacktestRequest):
    strat_path = Path(f"/strategies/candidates/{req.strategy_name}.py")
    strat_path.parent.mkdir(parents=True, exist_ok=True)
    strat_path.write_text(req.strategy_code)

    if req.phase == 1:
        stats = run_fast_filter(req.strategy_name)
        return {"phase": 1, "stats": stats, "passed": meets_phase1_criteria(stats)}

    if req.phase == 2:
        result = walk_forward_test(req.strategy_name)
        return {"phase": 2, "passed": result["passed"], "windows": result["windows"]}

    raise HTTPException(400, "phase must be 1 or 2")

@app.get("/health")
def health():
    return {"ok": True}
```

```python
# trinity: agent calls tela backtest API
import requests

TELA_BACKTEST_API = "https://backtest.starbluesolutions.net"

def agent_request_backtest(strategy_name: str, strategy_code: str, phase: int) -> dict:
    resp = requests.post(
        f"{TELA_BACKTEST_API}/backtest",
        json={"strategy_name": strategy_name, "strategy_code": strategy_code, "phase": phase},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()
```

### Trinity — Persistent Services

```yaml
# trinity: trading/podman-compose.yml
version: "3"

services:
  postgres:
    image: postgres:16
    restart: unless-stopped
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    environment:
      - POSTGRES_DB=trading
      - POSTGRES_USER=trading
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    ports:
      - "127.0.0.1:5432:5432"

  freqtrade-live:
    image: freqtradeorg/freqtrade:stable
    restart: unless-stopped
    volumes:
      - ./strategies:/freqtrade/strategies:ro
      - ./config:/freqtrade/config:ro
      - ./logs:/freqtrade/logs
    ports:
      - "127.0.0.1:8080:8080"
    environment:
      - HYPERLIQUID_WALLET_ADDRESS=${HYPERLIQUID_WALLET_ADDRESS}
      - HYPERLIQUID_PRIVATE_KEY=${HYPERLIQUID_PRIVATE_KEY}
    command: >
      trade
      --config /freqtrade/config/live.json
      --strategy LiveStrategy
      --logfile /freqtrade/logs/freqtrade.log

volumes:
  postgres_data:
```

### Folder Structure

```
# tela: ~/trading/
├── podman-compose.yml
├── backtest_api/
│   ├── main.py
│   ├── fast_filter.py          # Phase 1: numpy signal quality metrics
│   └── walk_forward.py
├── strategies/
│   └── candidates/               # written by agent via API
├── data/                         # local OHLCV feather copy (tela downloads own)
├── config/
│   └── backtest.json
└── backtest_results/

# trinity: ~/trading/
├── podman-compose.yml
├── strategies/
│   ├── LiveStrategy.py           # promoted winner
│   └── candidates/               # synced from tela post-promotion
├── data/                         # canonical OHLCV feather store
├── config/
│   ├── live.json                 # dry_run: false
│   └── paper/                   # per-candidate paper configs
├── paper/                        # paper arena orchestrator
│   └── orchestrator.py
├── sql/
│   └── init.sql                  # Postgres schema
├── .env                          # secrets (never commit)
└── logs/
```

---

## Freqtrade Config (Hyperliquid)

```json
{
  "exchange": {
    "name": "hyperliquid",
    "walletAddress": "${HYPERLIQUID_WALLET_ADDRESS}",
    "privateKey": "${HYPERLIQUID_PRIVATE_KEY}",
    "options": {
      "defaultType": "swap"
    },
    "pair_whitelist": [
      "BTC/USDC:USDC",
      "ETH/USDC:USDC",
      "SOL/USDC:USDC"
    ],
    "pair_blacklist": []
  },
  "trading_mode": "futures",
  "margin_mode": "isolated",
  "stake_currency": "USDC",
  "stake_amount": 33,
  "max_open_trades": 3,
  "timeframe": "4h",
  "dry_run": true,
  "dry_run_wallet": 100,
  "entry_pricing": {
    "price_side": "same",
    "use_order_book": true,
    "order_book_top": 1,
    "price_last_balance": 0.0,
    "check_depth_of_market": {"enabled": false, "bids_to_ask_delta": 1}
  },
  "exit_pricing": {
    "price_side": "other",
    "use_order_book": true,
    "order_book_top": 1
  },
  "pairlists": [{"method": "StaticPairList"}],
  "order_types": {
    "entry": "limit",
    "exit": "limit",
    "stoploss": "market",
    "stoploss_on_exchange": false
  },
  "api_server": {
    "enabled": true,
    "listen_ip_address": "0.0.0.0",
    "listen_port": 8080,
    "username": "freqtrade",
    "password": "${FREQTRADE_API_PASSWORD}",
    "jwt_secret_key": "${FREQTRADE_JWT_SECRET}"
  }
}
```

> **Notes:**
> - `"privateKey"` — capital K required. Lowercase silently passes `None` to ccxt, crashes at first order.
> - `pair_whitelist` lives under `exchange`, NOT top-level — Freqtrade ignores the top-level key.
> - `entry_pricing`, `exit_pricing`, `pairlists`, `order_types` are all required — Freqtrade will refuse to start without them.
> - Use `--strategy NullStrategy` at startup until a real strategy is promoted.

---

## Authentication — Hyperliquid Agent Wallet

Hyperliquid uses wallet private key signing (not API key/secret).

**Recommended: Agent Wallet (not your main wallet)**

```bash
# Generate a dedicated agent wallet
uv run --with eth-account python3 -c "
from eth_account import Account
acct = Account.create()
print('Address:', acct.address)
print('Private key:', acct.key.hex())
"
```

1. Generate fresh wallet
2. Fund it with USDC via Hyperliquid bridge
3. In the Hyperliquid UI, approve this address as an Agent for your main account
4. Use this agent key in config — if server is compromised, blast radius is limited to the agent wallet balance

---

## Strategy Format

Strategies have two forms: a lightweight **signal function** for Phase 1 fast filtering, and a full **Freqtrade strategy class** for Phase 2+ validation and live trading. Agents write both.

### Phase 1 — Signal Function (fast filter)

A plain Python function. No classes, no framework imports. The agent writes `generate_signals()` which returns a position Series (1 = long, 0 = flat, -1 = short).

```python
# strategies/candidates/GaussianChannel.py (Phase 1 format)

import pandas as pd
import pandas_ta as ta

def generate_signals(df: pd.DataFrame) -> pd.Series:
    """Return position series: 1=long, 0=flat, -1=short."""
    rsi = ta.rsi(df['close'], length=14)
    gc_upper = ta.ema(df['close'], length=20) + 2 * ta.stdev(df['close'], length=20)
    gc_lower = ta.ema(df['close'], length=20) - 2 * ta.stdev(df['close'], length=20)

    signals = pd.Series(0, index=df.index)
    signals[(df['close'] > gc_upper) & (rsi > 50) & (df['volume'] > 0)] = 1
    signals[df['close'] < gc_lower] = 0  # exit
    return signals
```

### Phase 2+ — Freqtrade Strategy Class (validation and live)

Once a signal function passes Phase 1, the agent translates it to a full Freqtrade strategy class for realistic backtesting and eventual live trading.

```python
# strategies/GaussianChannel.py (Freqtrade format)

from freqtrade.strategy import IStrategy
import pandas_ta as ta
import pandas as pd

class GaussianChannel(IStrategy):

    timeframe = '4h'
    minimal_roi = {"0": 0.15}
    stoploss = -0.08
    trailing_stop = True

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['gc_upper'] = ta.ema(df['close'], length=20) + 2 * ta.stdev(df['close'], length=20)
        df['gc_lower'] = ta.ema(df['close'], length=20) - 2 * ta.stdev(df['close'], length=20)
        return df

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df.loc[
            (df['close'] > df['gc_upper']) &
            (df['rsi'] > 50) &
            (df['volume'] > 0),
            'enter_long'
        ] = 1
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df.loc[
            (df['close'] < df['gc_lower']),
            'exit_long'
        ] = 1
        return df
```

---

## Backtesting Pipeline

### Phase 1 — Fast Iteration (Custom numpy/pandas)

Used by agent during strategy development loop. Sub-millisecond per evaluation — 10,000 variants in under 10 seconds. Zero framework dependencies.

The agent writes a **signal function** that takes a DataFrame and returns a Series of positions (1 = long, 0 = flat, -1 = short). A small `fast_filter.py` module handles all the accounting.

```python
# backtest_api/fast_filter.py

import numpy as np
import pandas as pd
from pathlib import Path

TAKER_FEE = 0.00045   # Hyperliquid taker rate (conservative — maker is 0.00015)
FUNDING_RATE_PER_BAR = 0.0001  # ~0.01%/hr synthetic drag, applied per candle close

def run_fast_filter(strategy_name: str) -> dict:
    """Load strategy module, run signal function on OHLCV data, compute metrics."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        strategy_name, f"/strategies/candidates/{strategy_name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Load OHLCV data (feather files, one per pair)
    pairs = ["BTC_USDC-USDC_4h", "ETH_USDC-USDC_4h", "SOL_USDC-USDC_4h"]
    results = {}
    for pair in pairs:
        df = pd.read_feather(f"/data/{pair}.feather")
        signals = mod.generate_signals(df)  # agent writes this function
        results[pair] = _compute_metrics(df["close"], signals)

    # Aggregate across pairs
    return _aggregate_metrics(results)


def _compute_metrics(close: pd.Series, signals: pd.Series) -> dict:
    """Compute signal quality metrics from a position series."""
    # Returns per bar (shifted to avoid lookahead)
    pos = signals.shift(1).fillna(0)
    returns = pos * close.pct_change()

    # Apply round-trip fee on each position change
    trades_mask = pos.diff().abs() > 0
    returns[trades_mask] -= TAKER_FEE

    # Apply synthetic funding drag on held positions (approximates Hyperliquid hourly funding)
    # KNOWN LIMITATION: Freqtrade Phase 2 does not model Hyperliquid hourly funding
    held_mask = pos.abs() > 0
    returns[held_mask] -= FUNDING_RATE_PER_BAR

    # Cumulative equity curve
    equity = (1 + returns).cumprod()

    # Sharpe (annualized for 4h bars — 6 bars/day × 365 days)
    bars_per_year = 6 * 365
    sharpe = (returns.mean() / returns.std()) * np.sqrt(bars_per_year) if returns.std() > 0 else 0

    # Max drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_drawdown = drawdown.min()

    # Trade segmentation (each continuous position is a trade)
    trade_boundaries = pos.diff().fillna(0).abs() > 0
    trade_ids = trade_boundaries.cumsum() * (pos != 0)
    trade_returns = returns.groupby(trade_ids).sum()
    trade_returns = trade_returns[trade_returns.index > 0]  # drop flat periods

    win_count = (trade_returns > 0).sum()
    loss_count = (trade_returns <= 0).sum()
    trade_count = len(trade_returns)
    win_rate = win_count / trade_count if trade_count > 0 else 0

    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = abs(trade_returns[trade_returns <= 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_return = equity.iloc[-1] - 1 if len(equity) > 0 else 0

    # Calmar (annualized return / max drawdown)
    years = len(close) / bars_per_year
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        "total_return":  round(total_return, 4),
        "sharpe":        round(sharpe, 4),
        "max_drawdown":  round(max_drawdown, 4),
        "win_rate":      round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "trade_count":   trade_count,
        "calmar":        round(calmar, 4),
    }


def _aggregate_metrics(results: dict) -> dict:
    """Average metrics across pairs."""
    keys = ["total_return", "sharpe", "max_drawdown", "win_rate", "profit_factor", "calmar"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in results.values() if isinstance(r[k], (int, float)) and r[k] != float("inf")]
        agg[k] = round(np.mean(vals), 4) if vals else 0
    agg["trade_count"] = sum(r["trade_count"] for r in results.values())
    agg["per_pair"] = results
    return agg


def meets_phase1_criteria(stats: dict) -> bool:
    return (
        stats["total_return"]  > 0.20   and
        stats["max_drawdown"]  > -0.20  and
        stats["profit_factor"] > 1.30   and
        stats["win_rate"]      > 0.45   and
        stats["trade_count"]   > 30     and
        stats["sharpe"]        > 0.80
    )
```

### Phase 2 — Realistic Validation (Freqtrade)

Only strategies that pass Phase 1 get here.

```python
import subprocess, json
from pathlib import Path

def run_freqtrade_backtest(strategy_name, timerange="20230101-20250101"):
    subprocess.run([
        "podman", "run", "--rm",
        "-v", "./strategies:/freqtrade/strategies",
        "-v", "./data:/freqtrade/user_data/data",
        "-v", "./backtest_results:/freqtrade/user_data/backtest_results",
        "-v", "./config:/freqtrade/config",
        "freqtradeorg/freqtrade:stable",
        "backtesting",
        "--config", "/freqtrade/config/config.json",
        "--strategy", strategy_name,
        "--timerange", timerange,
        "--export", "trades",
        "--export-filename", f"/freqtrade/user_data/backtest_results/{strategy_name}.json"
    ], check=True)

    results_file = Path(f"./backtest_results/{strategy_name}.json")
    return json.loads(results_file.read_text()) if results_file.exists() else None
```

### Walk-Forward Validation (Overfitting Prevention)

Strategy must perform across all three windows — not just the one it was tuned on.

```
Full dataset: 2023-01-01 → 2026-01-01 (3 years)

├── In-sample:      2023-01-01 → 2024-06-01  (tune here)
├── Validation:     2024-06-01 → 2025-06-01  (test here)
└── Out-of-sample:  2025-06-01 → 2026-01-01  (final gate)
```

```python
WINDOWS = [
    ("20230101", "20240601", "in-sample"),
    ("20240601", "20250601", "validation"),
    ("20250601", "20260101", "out-of-sample"),
]

def walk_forward_test(strategy_name):
    for start, end, label in WINDOWS:
        stats = run_freqtrade_backtest(strategy_name, f"{start}-{end}")
        pf = stats.get("profit_factor", 0)
        dd = stats.get("max_drawdown", -1)
        print(f"  {label}: PF={pf:.2f} DD={dd:.1%}")
        if pf < 1.2 or dd < -0.25:
            print(f"  ✗ Failed on {label} — reject")
            return False
    print("  ✓ Passed all windows")
    return True
```

---

## Phase 2.5 — Parallel Paper Trading

Strategies that pass walk-forward validation enter a paper trading arena before going live. Multiple Freqtrade instances run simultaneously in `dry_run: true` against live market data, with results accumulated in Postgres. The agent monitors all instances and promotes the winner.

### Why a Separate Paper Phase
- Backtest validation catches overfitting, but can't catch live execution surprises (spread, timing, slippage on illiquid pairs)
- Paper trading on live data with multiple candidates simultaneously surfaces relative performance without real capital risk
- Evaluation window is configurable — default 14 days

### Container Orchestration

Each candidate gets its own Freqtrade container with a unique port and isolated Postgres schema. The orchestrator spawns and tears them down dynamically — not static Compose services.

```python
# trading/paper/orchestrator.py

import subprocess, json, time, requests
from pathlib import Path
from dataclasses import dataclass

BASE_PORT = 8090   # 8090, 8091, 8092 ... per candidate

@dataclass
class PaperInstance:
    strategy_name: str
    port: int
    container_name: str
    db_schema: str

def spawn_paper_instance(strategy_name: str, slot: int) -> PaperInstance:
    port = BASE_PORT + slot
    container_name = f"paper_{strategy_name.lower()}_{slot}"
    db_schema = f"paper_{strategy_name.lower()}"

    # Write per-candidate config (inherits base config, overrides port + db)
    config = _build_paper_config(strategy_name, port, db_schema)
    config_path = Path(f"paper/configs/{container_name}.json")
    config_path.write_text(json.dumps(config, indent=2))

    subprocess.run([
        "podman", "run", "-d",
        "--name", container_name,
        "-v", f"./strategies:/freqtrade/strategies:ro",
        "-v", f"./data:/freqtrade/user_data/data:ro",
        "-v", f"./paper/configs:/freqtrade/config:ro",
        "-v", f"./logs:/freqtrade/logs",
        "-p", f"{port}:{port}",
        "--network", "trading_net",
        "freqtradeorg/freqtrade:stable",
        "trade",
        "--config", f"/freqtrade/config/{container_name}.json",
        "--strategy", strategy_name,
    ], check=True)

    return PaperInstance(strategy_name, port, container_name, db_schema)


def _build_paper_config(strategy_name: str, port: int, db_schema: str) -> dict:
    return {
        "exchange": {"name": "hyperliquid",
                     "walletAddress": "${HYPERLIQUID_WALLET_ADDRESS}",
                     "privateKey": "${HYPERLIQUID_PRIVATE_KEY}",
                     "options": {"defaultType": "swap"}},
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "stake_currency": "USDC",
        "stake_amount": 33,
        "max_open_trades": 3,
        "timeframe": "4h",
        "pair_whitelist": ["BTC/USDC:USDC", "ETH/USDC:USDC", "SOL/USDC:USDC"],
        "dry_run": True,
        "dry_run_wallet": 100,
        "db_url": f"postgresql://freqtrade:freqtrade@postgres:5432/freqtrade?options=-c search_path={db_schema}",
        "api_server": {
            "enabled": True,
            "listen_ip_address": "0.0.0.0",
            "listen_port": port,
            "username": "freqtrade",
            "password": "changeme",
            "jwt_secret_key": "generate-a-real-secret-here"
        }
    }


def teardown_paper_instance(instance: PaperInstance):
    subprocess.run(["podman", "stop", instance.container_name], check=False)
    subprocess.run(["podman", "rm",   instance.container_name], check=False)
    Path(f"paper/configs/{instance.container_name}.json").unlink(missing_ok=True)
```

### Agent Monitoring Loop

```python
# trading/paper/monitor.py

import requests, time, psycopg2
from typing import List

EVAL_WINDOW_DAYS = 14
POLL_INTERVAL_SECS = 3600  # check every hour

PROMOTION_CRITERIA = {
    "min_trades":        20,
    "min_profit_pct":    5.0,    # % total return over window
    "max_drawdown_pct": -15.0,
    "min_win_rate":      0.45,
    "min_profit_factor": 1.25,
}

def collect_metrics(instance: "PaperInstance") -> dict:
    base = f"http://localhost:{instance.port}"
    auth = ("freqtrade", "changeme")
    profit  = requests.get(f"{base}/api/v1/profit",      auth=auth).json()
    perf    = requests.get(f"{base}/api/v1/performance",  auth=auth).json()
    status  = requests.get(f"{base}/api/v1/status",       auth=auth).json()
    return {
        "strategy":       instance.strategy_name,
        "profit_pct":     profit.get("profit_all_percent", 0),
        "trade_count":    profit.get("trade_count", 0),
        "win_rate":       profit.get("winrate", 0),
        "profit_factor":  profit.get("profit_factor", 0),
        "max_drawdown":   profit.get("max_drawdown", 0),
        "open_trades":    len(status),
    }


def meets_promotion_criteria(metrics: dict) -> bool:
    c = PROMOTION_CRITERIA
    return (
        metrics["trade_count"]   >= c["min_trades"]        and
        metrics["profit_pct"]    >= c["min_profit_pct"]    and
        metrics["max_drawdown"]  >= c["max_drawdown_pct"]  and
        metrics["win_rate"]      >= c["min_win_rate"]      and
        metrics["profit_factor"] >= c["min_profit_factor"]
    )


def run_paper_arena(instances: List["PaperInstance"], eval_days: int = EVAL_WINDOW_DAYS):
    """
    Poll all paper instances every hour for eval_days.
    Return the best performer that meets criteria, or None.
    """
    deadline = time.time() + eval_days * 86400
    best = None

    while time.time() < deadline:
        all_metrics = [collect_metrics(i) for i in instances]
        # Persist snapshot to Postgres for later analysis
        _write_metrics_snapshot(all_metrics)

        candidates = [m for m in all_metrics if meets_promotion_criteria(m)]
        if candidates:
            best = max(candidates, key=lambda m: m["profit_factor"])
            print(f"  → Promotion candidate: {best['strategy']} "
                  f"PF={best['profit_factor']:.2f} "
                  f"WR={best['win_rate']:.1%} "
                  f"Return={best['profit_pct']:.1f}%")

        time.sleep(POLL_INTERVAL_SECS)

    return best


def _write_metrics_snapshot(all_metrics: list):
    conn = psycopg2.connect("postgresql://freqtrade:freqtrade@postgres:5432/freqtrade")
    with conn.cursor() as cur:
        for m in all_metrics:
            cur.execute("""
                INSERT INTO paper_snapshots
                  (ts, strategy, profit_pct, trade_count, win_rate, profit_factor, max_drawdown)
                VALUES (NOW(), %(strategy)s, %(profit_pct)s, %(trade_count)s,
                        %(win_rate)s, %(profit_factor)s, %(max_drawdown)s)
            """, m)
    conn.commit()
    conn.close()
```

### Postgres Schema

```sql
-- paper trading metrics history
CREATE TABLE paper_snapshots (
    id            SERIAL PRIMARY KEY,
    ts            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    strategy      TEXT NOT NULL,
    profit_pct    NUMERIC,
    trade_count   INTEGER,
    win_rate      NUMERIC,
    profit_factor NUMERIC,
    max_drawdown  NUMERIC
);

-- strategy registry: full lifecycle from candidate to live
CREATE TABLE strategy_registry (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    parent_strategy TEXT,
    iteration       INTEGER,
    phase1_passed   BOOLEAN,
    phase2_passed   BOOLEAN,
    paper_passed    BOOLEAN,
    promoted_live   BOOLEAN DEFAULT FALSE,
    promoted_at     TIMESTAMPTZ,
    notes           TEXT
);
```

### Promotion to Live

When a paper winner is found, the agent:
1. Stops all other paper instances
2. Copies the winning strategy to `strategies/LiveStrategy.py`
3. Calls `POST /api/v1/reload_config` on the live bot (or restarts it)
4. Updates `strategy_registry` row with `promoted_live = true`

```python
def promote_paper_winner(winner_instance: "PaperInstance", live_bot_port: int = 8080):
    import shutil
    src = f"strategies/candidates/{winner_instance.strategy_name}.py"
    shutil.copy(src, "strategies/LiveStrategy.py")

    auth = ("freqtrade", "changeme")
    requests.post(f"http://localhost:{live_bot_port}/api/v1/reload_config", auth=auth)

    # Update registry
    conn = psycopg2.connect("postgresql://freqtrade:freqtrade@postgres:5432/freqtrade")
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE strategy_registry
               SET paper_passed = true, promoted_live = true, promoted_at = NOW()
             WHERE name = %s
        """, (winner_instance.strategy_name,))
    conn.commit()
    conn.close()

    print(f"  ✓ {winner_instance.strategy_name} promoted to live")
```

### Resource Considerations (Trinity)

Trinity has 4 cores and 32GB RAM. Each Freqtrade instance is lightweight (~150-200MB RAM idle). Practical limit is **4-6 simultaneous paper instances** before CPU contention becomes an issue on a 4h timeframe strategy (low trade frequency, mostly sleeping).

---

## Agent Development Loop

```python
# Full strategy iteration loop

def agent_development_loop(initial_strategy_code, max_iterations=50):
    strategy_code = initial_strategy_code

    for i in range(max_iterations):
        # 1. Write strategy to disk
        strategy_name = f"Candidate{i:03d}"
        Path(f"strategies/candidates/{strategy_name}.py").write_text(strategy_code)

        # 2. Phase 1 — fast filter (custom numpy, sub-ms per eval)
        stats = run_fast_filter(strategy_name)
        print(f"[{i}] PF={stats['profit_factor']:.2f} "
              f"DD={stats['max_drawdown']:.1%} "
              f"SR={stats['sharpe']:.2f}")

        if meets_phase1_criteria(stats):
            print(f"  → Phase 1 passed, running Freqtrade validation...")

            # 3. Phase 2 — realistic validation
            if walk_forward_test(strategy_name):
                print(f"  → PROMOTING {strategy_name} to live")
                promote_to_live(strategy_name)
                return strategy_name

        # 4. Agent improves based on what failed
        strategy_code = agent_improve(strategy_code, stats)

    print("No strategy promoted within iteration budget")
    return None

def promote_to_live(strategy_name):
    # Copy to live strategies folder
    import shutil
    shutil.copy(
        f"strategies/candidates/{strategy_name}.py",
        f"strategies/LiveStrategy.py"
    )
    # Trigger Freqtrade config reload via REST
    import requests
    requests.post("http://localhost:8080/api/v1/reload_config",
                  auth=("freqtrade", "changeme"))
```

---

## Agent Monitoring (REST API)

```python
import requests

BASE = "http://your-podman-server:8080"
AUTH = ("freqtrade", "changeme")

# Status
requests.get(f"{BASE}/api/v1/status", auth=AUTH).json()

# Open trades
requests.get(f"{BASE}/api/v1/trades", auth=AUTH).json()

# P&L summary
requests.get(f"{BASE}/api/v1/profit", auth=AUTH).json()

# Performance by strategy
requests.get(f"{BASE}/api/v1/performance", auth=AUTH).json()

# Stop accepting new trades (keep open ones)
requests.post(f"{BASE}/api/v1/stopbuy", auth=AUTH)

# Full stop
requests.post(f"{BASE}/api/v1/stop", auth=AUTH)

# Force exit a position
requests.post(f"{BASE}/api/v1/forceexit",
    json={"tradeid": "1", "ordertype": "market"}, auth=AUTH)

# Hot reload config/strategy
requests.post(f"{BASE}/api/v1/reload_config", auth=AUTH)
```

---

## Data Layer

### Data Accumulation (START IMMEDIATELY)

**Critical constraint:** Hyperliquid's API returns a maximum of **5,000 candles per request**. At 4h, that's ~833 days (~2.3 years). The design doc's 3-year timerange (20230101-20260101 = ~6,570 candles at 4h) exceeds this limit. Data must be accumulated incrementally over time — start the collection job on day one.

```python
# scripts/download_data.py

import ccxt
import pandas as pd
from pathlib import Path

def download_hyperliquid_data(pairs, timeframes, since="2023-01-01"):
    hl = ccxt.hyperliquid({'options': {'defaultType': 'swap'}})
    Path("/data").mkdir(exist_ok=True)

    for pair in pairs:
        for tf in timeframes:
            print(f"Downloading {pair} {tf}...")
            ohlcv = hl.fetch_ohlcv(
                pair, tf,
                since=hl.parse8601(f"{since}T00:00:00Z"),
                limit=5000    # hard API max — cannot request more
            )
            df = pd.DataFrame(ohlcv,
                columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            fname = f"/data/{pair.replace('/','_').replace(':','-')}_{tf}.feather"

            # Append to existing data if present
            if Path(fname).exists():
                existing = pd.read_feather(fname)
                df = pd.concat([existing, df]).drop_duplicates().sort_index()

            df.to_feather(fname)
            print(f"  ✓ {len(df)} candles → {fname}")

download_hyperliquid_data(
    pairs=["BTC/USDC:USDC", "ETH/USDC:USDC", "SOL/USDC:USDC"],
    timeframes=["1h", "4h", "1d"],
    since="2023-01-01"
)
```

### Refresh Data (run daily via cron)

```bash
# Add to cron — incremental append, not full re-download
podman run --rm \
  -v ./data:/freqtrade/user_data/data \
  -v ./config:/freqtrade/config \
  freqtradeorg/freqtrade:stable \
  download-data \
  --config /freqtrade/config/config.json \
  --exchange hyperliquid \
  --trading-mode futures \
  --pairs BTC/USDC:USDC ETH/USDC:USDC SOL/USDC:USDC \
  --timeframes 1h 4h 1d \
  --days 30
```

**Note:** `freqtrade download-data` also has the 5,000-candle cap. `--dl-trades` is explicitly unsupported for Hyperliquid. Mark candles are unavailable (Freqtrade uses regular candles instead as of 2025.11).

---

## IAM / Security

Kanidm is already deployed on infrastructure.

- Self-hosted, single Rust binary, ~50-100MB RAM
- OAuth2/OIDC for interactive browser login (UIs)
- Service account API tokens for machine-to-machine (via RFC 8693 token exchange — no standard `client_credentials` grant)
- No external DB required (embedded)
- **Note:** Kanidm requires TLS even as a backend — cannot sit behind a plain HTTP reverse proxy
- Services that don't speak OIDC natively (Freqtrade) need **oauth2-proxy** in front

**Protects:**
- Freqtrade REST API (8080) — via oauth2-proxy
- Backtest API on tela — via oauth2-proxy or Caddy basicauth (simpler for internal)
- Any future dashboards/UIs

**Agent auth flow:**
1. Create Kanidm service account for OpenClaw
2. Generate API token: `kanidm service-account api-token generate`
3. Agent exchanges API token for OAuth2 JWT via `POST /oauth2/token` (RFC 8693 token exchange)
4. JWT used as `Authorization: Bearer` against protected APIs

---

## Key Confirmed Facts (verified March 28, 2026)

- ✅ `ccxt.hyperliquid` exists and is fully functional
- ✅ `fetchOHLCV`, `createOrder`, `fetchPositions`, `setLeverage`, `cancelOrder` all supported
- ✅ 378 perp markets, 284 spot markets available
- ✅ Perps settle in USDC, isolated + cross margin supported
- ✅ Freqtrade has `hyperliquid` in its exchange constants (stable since 2025.3, current 2026.2)
- ✅ `pandas-ta` covers all major indicators (Gaussian Channel, Supertrend, Ichimoku, etc.)
- ✅ Agent wallets supported: `walletAddress` (master) + `privateKey` (agent) — agent can trade but not withdraw

## Known Caveats and Gotchas

### Hyperliquid Exchange
- **No market orders** — ccxt simulates with limit orders at 5% max slippage. If slippage exceeds 5%, order is rejected.
- **Funding rates settle hourly** (not 8h like Binance), capped at 4%/hr. For strategies holding positions hours-to-days, this is material cost. Freqtrade does not reliably model Hyperliquid funding in backtests — treat Phase 2 backtest results as optimistic for hold-duration strategies.
- **5,000 candle API limit** — cannot bulk-download years of history. Must accumulate incrementally.
- **HIP-3 pairs** (builder-deployed perps, e.g. tokenized equities) have different naming conventions — use `freqtrade list-pairs --exchange hyperliquid --trading-mode futures` to get exact names.
- **$10 minimum order** notional value across all pairs.
- **Max leverage:** 50x for BTC/ETH perps.

### Freqtrade Integration
- **`privateKey` capitalization trap** — must be capital K. Lowercase silently passes `None` to ccxt and crashes at first order.
- **Rate limits** — Hyperliquid enforces 10,000 points/address. `stoploss_on_exchange` burns points via `set_margin_mode()` calls. Keep `stoploss_on_exchange_interval` at 60s+.
- **Add `coincurve` to container** — drops ECDSA signing from ~45ms to <0.05ms per order. `pip install coincurve` in the Freqtrade Dockerfile.
- **Use a named agent wallet** — unnamed agents get deregistered when a new one is created, pruning nonce state.
- **Exclusive account access required** — any external trading on the same account desyncs Freqtrade's internal state.
- **Mark candles unavailable** — Freqtrade uses regular candles (fixed in 2025.11).
- **`--dl-trades` unsupported** for Hyperliquid.

---

## Rollout Order

**Data (start immediately — accumulation is the long pole):**
0. **Start OHLCV collection** on both tela and trinity — daily cron, append mode. The 5,000-candle limit means 4h data needs weeks of accumulation to reach 3-year coverage.

**Tela setup:**
1. **Backtest API** — stand up FastAPI service with `fast_filter.py`, wire Caddy → `backtest.starbluesolutions.net`
2. **Phase 1 smoke test** — send a known signal function via API, confirm numpy metrics return correctly
3. **Phase 2 smoke test** — confirm Freqtrade walk-forward runs via API

**Trinity setup:**
4. **Postgres** — stand up container, run init.sql (strategy_registry, paper_snapshots)
5. **Freqtrade dry run** — single instance, validate Hyperliquid connection + REST API (install `coincurve`)

**Wiring:**
6. **Agent loop** — OpenClaw writes signal functions, calls tela API Phase 1 → Phase 2, stores results in Postgres
7. **Paper arena** — orchestrator spawns N parallel paper containers on trinity, monitor loop runs
8. **Kanidm integration** — secure the REST APIs and any UIs before going live (Kanidm already deployed)
9. **Live** — fund named agent wallet, promote paper winner, flip `dry_run: false`, start small

---

## Resolved Decisions (2026-03-28)

- [x] **TLS for backtest API on tela** — Caddy, same pattern as sentinel/embed.api → `backtest.starbluesolutions.net`. Added to starblue-infra `caddy_gpu_dev` service list.
- [x] **TLS for Freqtrade REST on trinity** — Tailscale-only, no public exposure. Caddy proxy on trinity for TLS termination within mesh. Freqtrade binds to `127.0.0.1:8080`.
- [x] **Starting capital** — $100 USDC. `stake_amount: 33` (3 max open trades × $33 ≈ $100).
- [x] **Pair list** — BTC/ETH/SOL as crypto baseline, plus HIP-3 real-world assets: S&P 500, gold, WTI crude (see "HIP-3 Assets" section below). Use `freqtrade list-pairs` to get exact HIP-3 pair names.
- [x] **Leverage** — 1x (spot-like). Increase only after paper arena validation at 1x.
- [x] **Paper arena eval window** — 14 days default, configurable via `EVAL_WINDOW_DAYS` env var.
- [x] **Max paper instances** — Dynamic based on trinity resources. Start with 4, monitor RAM/CPU. Each instance ~150-200MB.
- [x] **Paper arena port range** — 8090+ on trinity is safe. Trinity services: Jenkins 8081, OpenClaw 3000s, scrubclaw 8443/9090. No conflicts.
- [x] **Funding rate** — Add synthetic funding drag in Phase 1 numpy filter only (`FUNDING_RATE_PER_BAR = 0.0001`). Accept as known limitation in Phase 2 Freqtrade. At $100/1x, funding is ~$0.04/4h-hold — same order as one taker fee. Revisit if scaling beyond $10K or 3x leverage.
- [x] **Kanidm integration** — oauth2-proxy in front of Freqtrade REST. Agent auth via RFC 8693 token exchange (Kanidm service account → scoped JWT → Bearer token). Kanidm already supports this natively.
- [x] **HYPE staking** — Skip. No HYPE holdings. Fees stay at base rate (taker 0.045%, maker 0.015%).

## HIP-3 Real-World Assets

Hyperliquid now offers perpetual futures on equities, indices, and commodities via HIP-3 (permissionless perp deployment protocol). These are cash-settled perps with oracle-based pricing, identical mechanics to crypto perps. 24/7 trading including weekends.

**Available as of March 2026:**

| Category | Examples | Notes |
|---|---|---|
| Equity indices | S&P 500, NASDAQ 100, MAG7 basket | Licensed from S&P DJI, launched 2026-03-18 |
| Commodities | WTI crude oil, Brent crude, gold, silver | High volume — WTI hit $1.27B/day |
| Individual stocks | 260+ U.S. stocks/ETFs via Felix/Ondo | These are HyperEVM spot tokens, NOT HIP-3 perps — different mechanism |

**Target HIP-3 pairs for this stack:**
- **Crypto:** BTC/USDC:USDC, ETH/USDC:USDC, SOL/USDC:USDC
- **Indices:** S&P 500 perp (exact pair name TBD via `freqtrade list-pairs`)
- **Commodities:** Gold perp, WTI crude perp (exact pair names TBD)

**Caveats:**
- HIP-3 pair naming may differ from standard crypto pairs — always use `freqtrade list-pairs` to confirm
- Liquidity and spread on newer HIP-3 pairs may be thinner than BTC/ETH
- Oracle-based pricing means no orderbook for some pairs — check before trading
- Start with crypto baseline, add HIP-3 pairs after confirming Freqtrade compatibility

---

---

## Implementation Status

### Completed

| Step | What | Status |
|------|------|--------|
| 0 | Create `eddiedunn/trading` repo, seed OHLCV data | ✅ DONE |
| 1 | Full repo scaffolding — backtest_api, paper, sql, config, tests (13 pass) | ✅ DONE |
| CI/CD | Jenkins pipeline (Trivy + pytest + SonarQube), Dockerfile, Quadlet, auto-deploy | ✅ DONE |
| 2 | Backtest API live on tela — `https://backtest.starbluesolutions.net/health` | ✅ DONE |
| 3 | Trinity Ansible role — Postgres + Freqtrade Quadlets, crons, secrets | ✅ DONE |
| 4 | Trade skill in toolshed — SKILL.md, config.json, trading_client.py | ✅ DONE |

### Trinity Services (as of 2026-03-28)

- **postgres-trading**: active (rootless Podman, port 5432, Postgres 16)
- **freqtrade-live**: active (rootless Podman, port 8080, Freqtrade 2026.2, dry_run=true, NullStrategy)
- Crons: OHLCV download 1am, paper cleanup 1:30am

### Pending

| Step | What | Blocked by |
|------|------|-----------|
| 5 | Phase 1 e2e: agent writes + backtests strategy | Nothing |
| 6 | Phase 2 e2e: walk-forward validation | Step 5 winner |
| 7 | Paper arena: spawn instances, monitor 14 days | Step 6 winner |
| 8 | Live trading: create Hyperliquid agent wallet, fund $100 USDC, promote winner | Step 7 winner + human APPROVE |

### Deployment Notes

**Freqtrade config required fields** (not in original template — added during deploy):
- `entry_pricing` and `exit_pricing` blocks — required by Freqtrade exchange validation
- `pairlists: [{method: StaticPairList}]` — required even when using `pair_whitelist`
- `order_types` block — required for futures trading
- `pair_whitelist` must be under `exchange` key, NOT top-level
- `dry_run_wallet: 100` — sets starting virtual balance in dry_run mode

**Rootless Podman + bind mounts**: Container user `ftuser` (uid 1001 inside container) maps to a subuid (~100001) on the host. Host directories bound into the container must be:
- Readable by all (`0644` for files, `0755` for dirs) if the container needs read access
- World-writable (`0777`) if the container needs write access — OR omit the bind mount and let Freqtrade use its internal path

**`--logfile` flag**: Causes a `PermissionError` in Freqtrade's logging config when the log directory is a host bind mount. Dropped — logs flow to journald via `journalctl --user -u freqtrade-live`.

**NullStrategy**: `strategies/NullStrategy.py` in trading repo. A no-op IStrategy that never opens positions. Required because `--strategy` is mandatory at Freqtrade startup. Replace with `live promote` when a real strategy is ready.

### gopass Secrets

| Path | Status |
|------|--------|
| `trading/postgres-password` | ✅ generated |
| `trading/freqtrade-api-password` | ✅ generated |
| `trading/freqtrade-jwt-secret` | ✅ generated |
| `trading/hyperliquid-wallet-address` | ⏳ pending (Step 8) |
| `trading/hyperliquid-private-key` | ⏳ pending (Step 8) |

---

*Document generated from design session — March 28, 2026*
*Updated March 28, 2026: replaced VectorBT with custom numpy/pandas, corrected fees/auth/data caveats*
*Updated March 28, 2026: Steps 0-4 complete, trinity services live, deployment notes added*
*Updated March 28, 2026: resolved all open questions, added HIP-3 real-world assets, synthetic funding drag, $100 capital / 1x leverage, registered in starblue-infra*
