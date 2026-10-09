#!/usr/bin/env python3
"""Candles and hourly funding for Hyperliquid HIP-3 (builder-deployed) markets, e.g. the
trade.xyz stock, index and commodity perps, straight from the info API.

Writes the same files scripts/download_data.py writes for crypto perps, so Phase 1 research
code can read them unchanged: <NAME>_USDC-USDC_<tf>.feather (timestamp open high low close
volume) and <NAME>_USDC-USDC_funding_1h.feather (timestamp rate, stamped on the hour).
<NAME> is the market name with the dex prefix folded in: "xyz:NVDA" -> "xyzNVDA".

The candle endpoint returns at most the newest 5000 bars per call; funding is paged.
Research only.

Usage:
    uv run python scripts/download_hip3.py                      # default trade.xyz markets
    uv run python scripts/download_hip3.py --coins xyz:NVDA xyz:GOLD
    uv run python scripts/download_hip3.py --list xyz           # markets on a dex by volume
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

INFO = "https://api.hyperliquid.xyz/info"
DEFAULT_COINS = ["xyz:SP500", "xyz:XYZ100", "xyz:GOLD", "xyz:SILVER", "xyz:CL", "xyz:NVDA", "xyz:MU", "xyz:SNDK", "xyz:SKHX", "xyz:TSLA"]
TIMEFRAMES = ["1h", "4h", "1d"]


def post(payload: dict):
    for attempt in range(6):
        r = requests.post(INFO, json=payload, timeout=60)
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"rate limited: {payload}")


def list_markets(dex: str) -> None:
    meta, ctxs = post({"type": "metaAndAssetCtxs", "dex": dex})
    rows = [(a["name"], float(c["dayNtlVlm"]) / 1e6, float(c["openInterest"]) * float(c["markPx"]) / 1e6)
            for a, c in zip(meta["universe"], ctxs) if not a.get("isDelisted")]
    df = pd.DataFrame(rows, columns=["market", "day_volume_M", "open_interest_M"]).sort_values("day_volume_M", ascending=False)
    print(df.to_string(index=False))


def candles(coin: str, tf: str) -> pd.DataFrame:
    raw = post({"type": "candleSnapshot", "req": {"coin": coin, "interval": tf, "startTime": 0, "endTime": int(time.time() * 1000)}})
    df = pd.DataFrame([{"timestamp": c["t"], "open": float(c["o"]), "high": float(c["h"]), "low": float(c["l"]),
                        "close": float(c["c"]), "volume": float(c["v"])} for c in raw])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype("datetime64[ms, UTC]")
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def funding(coin: str) -> pd.DataFrame:
    rows, since = [], 0
    while True:
        batch = post({"type": "fundingHistory", "coin": coin, "startTime": since})
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1]["time"] + 1
        if nxt == since or len(batch) < 500:
            break
        since = nxt
        time.sleep(0.5)
    df = pd.DataFrame([{"timestamp": r["time"], "rate": float(r["fundingRate"])} for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.floor("h")
    return df.groupby("timestamp", as_index=False)["rate"].sum()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    ap.add_argument("--coins", nargs="+", default=DEFAULT_COINS)
    ap.add_argument("--list", metavar="DEX")
    args = ap.parse_args(argv)
    if args.list:
        list_markets(args.list)
        return 0
    for coin in args.coins:
        name = coin.replace(":", "")
        for tf in TIMEFRAMES:
            df = candles(coin, tf)
            df.to_feather(args.data_dir / f"{name}_USDC-USDC_{tf}.feather")
            print(f"{coin} {tf}: {len(df)} bars {df['timestamp'].min():%Y-%m-%d} .. {df['timestamp'].max():%Y-%m-%d}")
            time.sleep(0.5)
        f = funding(coin)
        f.to_feather(args.data_dir / f"{name}_USDC-USDC_funding_1h.feather")
        print(f"{coin} funding: {len(f)} hours from {f['timestamp'].min():%Y-%m-%d}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
