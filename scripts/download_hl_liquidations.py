#!/usr/bin/env python3
"""
Hyperliquid liquidation fills from Hydromancer's Reservoir archive (requester-pays S3,
Parquet, Hyperliquid perps from August 2025). Reads each day's fills file with DuckDB
straight from S3, keeping only liquidation rows for the chosen coins and only the
columns we need, so egress is a small fraction of the file.

Writes ``<COIN>_USDC-USDC_liquidations.feather`` (one row per liquidation fill:
timestamp, side, price, size, direction, method, mark price, realized pnl, address) into
the data dir, and caches one Parquet per day under ``<data-dir>/hl_liquidations/``.

Needs: ``aws login`` (the script exports temporary credentials from the CLI), DuckDB
(``uv run --with duckdb``). Bucket region ap-northeast-1.

Usage:
    uv run --with duckdb python scripts/download_hl_liquidations.py
    uv run --with duckdb python scripts/download_hl_liquidations.py --since 2025-08-01 --coins BTC ETH SOL
    uv run --with duckdb python scripts/download_hl_liquidations.py --list   # show the bucket layout and stop
"""

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

BUCKET = "s3://hydromancer-reservoir"
REGION = "ap-northeast-1"
DEFAULT_PREFIX = "by_dex/hyperliquid/fills/perp/liquidations"  # liquidation fills only; the all/ files are ~1 GB a day
DEFAULT_SINCE = "2025-07-28"  # first day in the bucket
DEFAULT_COINS = ["BTC", "ETH", "SOL"]
COLS = ["timestamp", "base_symbol", "side", "price", "size", "direction", "liquidation_method",
        "liquidation_mark_px", "realized_pnl", "start_position", "address", "crossed"]


def aws_env_credentials() -> dict:
    r = subprocess.run(["aws", "configure", "export-credentials", "--format", "process"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"aws credentials unavailable: {r.stderr.strip()[:200]}  (run: aws login)")
    return json.loads(r.stdout)


def connect():
    import duckdb
    c = duckdb.connect()
    c.execute("INSTALL httpfs; LOAD httpfs;")
    cred = aws_env_credentials()
    c.execute(f"""CREATE SECRET s3 (TYPE S3, KEY_ID '{cred['AccessKeyId']}', SECRET '{cred['SecretAccessKey']}',
                  SESSION_TOKEN '{cred.get('SessionToken', '')}', REGION '{REGION}', REQUESTER_PAYS true)""")
    c.execute("SET s3_requester_pays = true")
    return c


def list_layout(prefix: str) -> None:
    for p in ("", "by_dex/", "by_dex/hyperliquid/", "by_dex/hyperliquid/fills/", "by_dex/hyperliquid/fills/perp/", prefix.rstrip("/") + "/"):
        r = subprocess.run(["aws", "s3", "ls", "--request-payer", "requester", f"{BUCKET}/{p}"], capture_output=True, text=True)
        print(f"== {BUCKET}/{p}\n" + (r.stdout.strip()[:1500] or r.stderr.strip()[:300]))


def fetch_day(c, prefix: str, day: date, coins: list[str], cache: Path) -> pd.DataFrame | None:
    out = cache / f"{day.isoformat()}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    url = f"{BUCKET}/{prefix}/date={day.isoformat()}/fills.parquet"
    coin_list = ", ".join(f"'{x}'" for x in coins)
    try:
        df = c.execute(f"""SELECT {', '.join(COLS)} FROM read_parquet('{url}')
                           WHERE is_liquidation = true AND base_symbol IN ({coin_list})""").df()
    except Exception as e:  # missing partition means no file for that day
        msg = str(e)
        if "404" in msg or "Not Found" in msg or "NoSuchKey" in msg or "HTTP 404" in msg:
            return None
        raise
    df.to_parquet(out, index=False)
    return df


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--since", default=DEFAULT_SINCE)
    ap.add_argument("--until", default=None)
    ap.add_argument("--coins", nargs="+", default=DEFAULT_COINS)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        list_layout(args.prefix)
        return 0
    cache = args.data_dir / "hl_liquidations"
    cache.mkdir(parents=True, exist_ok=True)
    c = connect()
    start = date.fromisoformat(args.since)
    end = date.fromisoformat(args.until) if args.until else datetime.now(timezone.utc).date() - timedelta(days=1)
    frames, missing = [], []
    day = start
    while day <= end:
        df = fetch_day(c, args.prefix, day, args.coins, cache)
        if df is None:
            missing.append(day)
        else:
            frames.append(df)
        day += timedelta(days=1)
    print(f"days with a file: {len(frames)}, days missing: {len(missing)}" + (f" (first few: {[d.isoformat() for d in missing[:5]]})" if missing else ""))
    if not frames:
        return 1
    allrows = pd.concat(frames, ignore_index=True)
    allrows["timestamp"] = pd.to_datetime(allrows["timestamp"], utc=True)
    for coin in args.coins:
        d = allrows[allrows["base_symbol"] == coin].sort_values("timestamp").reset_index(drop=True)
        out = args.data_dir / f"{coin}_USDC-USDC_liquidations.feather"
        d.to_feather(out)
        notional = (d["price"].astype(float) * d["size"].astype(float)).sum()
        print(f"{coin}: {len(d):,} liquidation fills {d['timestamp'].min():%Y-%m-%d} .. {d['timestamp'].max():%Y-%m-%d}, "
              f"${notional/1e6:,.0f}M notional, methods {d['liquidation_method'].value_counts().to_dict()} -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
