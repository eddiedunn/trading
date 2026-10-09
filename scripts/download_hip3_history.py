#!/usr/bin/env python3
"""Full hourly history for HIP-3 markets from Hydromancer's Reservoir mirror (requester-pays S3,
`by_dex/<dex>/candles/1s/date=YYYY-MM-DD/candles.parquet`, from 2025-10-13 for trade.xyz).

The info API only serves the newest 5000 candles, so this is the way to get 1h bars back to a
market's listing. Each day's one-second candles are aggregated to hourly inside the DuckDB read
(only the chosen coins' rows leave S3), cached as data/hip3_history/<dex>/<date>.parquet, then
written as <NAME>_USDC-USDC_1h.feather and _4h.feather in the crypto-perp format (volume in base
units), replacing the API-pulled candle files. Funding still comes from scripts/download_hip3.py.

Needs a live `aws login` session. Run with: uv run --with duckdb python scripts/download_hip3_history.py
"""
import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd

_EXPIRES = None

BUCKET = "hydromancer-reservoir"
REGION = "ap-northeast-1"
DEFAULT_COINS = ["xyz:SP500", "xyz:XYZ100", "xyz:GOLD", "xyz:SILVER", "xyz:CL", "xyz:NVDA", "xyz:MU",
                 "xyz:GOOGL", "xyz:META", "xyz:TSLA", "xyz:AMZN", "xyz:MSTR"]


def connect() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("INSTALL httpfs; LOAD httpfs; SET s3_requester_pays=true")
    refresh(c)
    return c


def refresh(c: duckdb.DuckDBPyConnection) -> None:
    """(Re)create the S3 secret from the CLI's current credentials. `aws login` hands out
    15-minute tokens that the CLI refreshes itself, so this is called again before each
    day's read once the token is within two minutes of expiry."""
    global _EXPIRES
    cred = json.loads(subprocess.check_output(["aws", "configure", "export-credentials", "--format", "process"]))
    exp = cred.get("Expiration")
    _EXPIRES = pd.Timestamp(exp) if exp else None
    c.execute("DROP SECRET IF EXISTS reservoir")
    c.execute(f"""CREATE SECRET reservoir (TYPE S3, KEY_ID '{cred['AccessKeyId']}', SECRET '{cred['SecretAccessKey']}',
                  SESSION_TOKEN '{cred.get('SessionToken', '')}', REGION '{REGION}')""")


def ensure_fresh(c: duckdb.DuckDBPyConnection) -> None:
    if _EXPIRES is not None and pd.Timestamp.now(tz="UTC") > _EXPIRES - pd.Timedelta(minutes=2):
        refresh(c)


def fetch_day(c, dex: str, day: date, coins: list[str], cache: Path) -> pd.DataFrame | None:
    out = cache / f"{day.isoformat()}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    ensure_fresh(c)
    url = f"s3://{BUCKET}/by_dex/{dex}/candles/1s/date={day.isoformat()}/candles.parquet"
    coin_list = ", ".join(f"'{x}'" for x in coins)
    try:
        df = c.execute(f"""
            SELECT coin, date_trunc('hour', timestamp) AS timestamp,
                   arg_min(open, timestamp)::DOUBLE AS open, max(high)::DOUBLE AS high, min(low)::DOUBLE AS low,
                   arg_max(close, timestamp)::DOUBLE AS close, sum(volume)::DOUBLE AS volume, sum(trade_count) AS trades
            FROM read_parquet('{url}') WHERE coin IN ({coin_list})
            GROUP BY coin, date_trunc('hour', timestamp)""").df()
    except duckdb.HTTPException as e:
        if "404" in str(e):
            return None
        raise
    df.to_parquet(out)
    return df


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    ap.add_argument("--dex", default="xyz")
    ap.add_argument("--since", default="2025-10-13")
    ap.add_argument("--until", default=None)
    ap.add_argument("--coins", nargs="+", default=DEFAULT_COINS)
    args = ap.parse_args(argv)
    cache = args.data_dir / "hip3_history" / args.dex
    cache.mkdir(parents=True, exist_ok=True)
    c = connect()
    start = date.fromisoformat(args.since)
    end = date.fromisoformat(args.until) if args.until else datetime.now(timezone.utc).date() - timedelta(days=1)
    frames, missing, day = [], [], start
    while day <= end:
        df = fetch_day(c, args.dex, day, args.coins, cache)
        (frames if df is not None else missing).append(df if df is not None else day)
        day += timedelta(days=1)
    print(f"days with a file: {len(frames)}, missing: {len(missing)}" + (f" (first few: {[d.isoformat() for d in missing[:5]]})" if missing else ""))
    if not frames:
        return 1
    allrows = pd.concat(frames, ignore_index=True)
    allrows["timestamp"] = pd.to_datetime(allrows["timestamp"], utc=True)
    for coin in args.coins:
        name = coin.replace(":", "")
        h = allrows[allrows["coin"] == coin].drop(columns=["coin", "trades"]).sort_values("timestamp").reset_index(drop=True)
        if h.empty:
            print(f"{coin}: no rows")
            continue
        h["timestamp"] = h["timestamp"].astype("datetime64[ms, UTC]")
        h.to_feather(args.data_dir / f"{name}_USDC-USDC_1h.feather")
        g = h.set_index("timestamp").resample("4h", label="left", closed="left").agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum")).dropna(subset=["close"]).reset_index()
        g["timestamp"] = g["timestamp"].astype("datetime64[ms, UTC]")
        g.to_feather(args.data_dir / f"{name}_USDC-USDC_4h.feather")
        print(f"{coin}: {len(h):,} hours {h['timestamp'].min():%Y-%m-%d} .. {h['timestamp'].max():%Y-%m-%d}, {len(g):,} 4h bars")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
