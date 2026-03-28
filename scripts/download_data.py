#!/usr/bin/env python3
"""
OHLCV data downloader for Hyperliquid perpetual futures.

Incrementally appends candles to feather files. Safe to run repeatedly —
deduplicates on timestamp. Designed to be called by daily cron.

Usage:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --pairs BTC/USDC:USDC ETH/USDC:USDC
    uv run python scripts/download_data.py --data-dir /custom/path
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

# Hyperliquid API returns max 5000 candles per request
API_CANDLE_LIMIT = 5000

DEFAULT_PAIRS = [
    "BTC/USDC:USDC",
    "ETH/USDC:USDC",
    "SOL/USDC:USDC",
]

DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]

# How far back to start if no existing data (ISO date)
DEFAULT_SINCE = "2023-01-01"


def pair_to_filename(pair: str, timeframe: str) -> str:
    """Convert pair like 'BTC/USDC:USDC' + '4h' to 'BTC_USDC-USDC_4h.feather'."""
    return f"{pair.replace('/', '_').replace(':', '-')}_{timeframe}.feather"


def download_pair(
    exchange: ccxt.Exchange,
    pair: str,
    timeframe: str,
    data_dir: Path,
    since: str = DEFAULT_SINCE,
) -> int:
    """Download OHLCV data for a single pair/timeframe. Returns candle count."""
    fname = data_dir / pair_to_filename(pair, timeframe)

    # Determine start time
    if fname.exists():
        existing = pd.read_feather(fname)
        if "timestamp" in existing.columns and len(existing) > 0:
            # Start from last candle timestamp to get overlap for dedup
            last_ts = existing["timestamp"].max()
            if isinstance(last_ts, pd.Timestamp):
                since_ms = int(last_ts.timestamp() * 1000)
            else:
                since_ms = int(last_ts)
        else:
            since_ms = exchange.parse8601(f"{since}T00:00:00Z")
            existing = pd.DataFrame()
    else:
        since_ms = exchange.parse8601(f"{since}T00:00:00Z")
        existing = pd.DataFrame()

    print(f"  Fetching {pair} {timeframe} since {datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).isoformat()}")

    # Paginate through candles (API returns max 5000 per call)
    all_candles = []
    current_since = since_ms

    while True:
        candles = exchange.fetch_ohlcv(
            pair, timeframe, since=current_since, limit=API_CANDLE_LIMIT
        )
        if not candles:
            break

        all_candles.extend(candles)

        # If we got fewer than the limit, we've reached the end
        if len(candles) < API_CANDLE_LIMIT:
            break

        # Move cursor past last candle
        current_since = candles[-1][0] + 1

    if not all_candles:
        print(f"    No new candles for {pair} {timeframe}")
        return len(existing) if not existing.empty else 0

    # Build DataFrame
    new_df = pd.DataFrame(
        all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True)

    # Merge with existing data, deduplicate
    if not existing.empty:
        if not pd.api.types.is_datetime64_any_dtype(existing["timestamp"]):
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        combined = pd.concat([existing, new_df])
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    combined = combined.reset_index(drop=True)

    # Write feather
    combined.to_feather(fname)
    added = len(combined) - (len(existing) if not existing.empty else 0)
    print(f"    {len(combined)} total candles ({added} new) -> {fname.name}")
    return len(combined)


def main():
    parser = argparse.ArgumentParser(description="Download Hyperliquid OHLCV data")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=DEFAULT_PAIRS,
        help="Trading pairs to download",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=DEFAULT_TIMEFRAMES,
        help="Timeframes to download",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Directory for feather files",
    )
    parser.add_argument(
        "--since",
        default=DEFAULT_SINCE,
        help="Start date for initial download (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Hyperliquid OHLCV Download — {datetime.now(timezone.utc).isoformat()}")
    print(f"  Pairs: {args.pairs}")
    print(f"  Timeframes: {args.timeframes}")
    print(f"  Data dir: {args.data_dir}")
    print()

    exchange = ccxt.hyperliquid({"options": {"defaultType": "swap"}})

    errors = []
    for pair in args.pairs:
        for tf in args.timeframes:
            try:
                download_pair(exchange, pair, tf, args.data_dir, args.since)
            except Exception as e:
                msg = f"ERROR downloading {pair} {tf}: {e}"
                print(f"    {msg}", file=sys.stderr)
                errors.append(msg)

    print()
    if errors:
        print(f"Completed with {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)
    else:
        print("All downloads complete.")


if __name__ == "__main__":
    main()
