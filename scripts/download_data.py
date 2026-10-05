#!/usr/bin/env python3
"""
OHLCV data downloader for Hyperliquid perpetual futures.

Incrementally appends candles to feather files. Safe to run repeatedly —
deduplicates on timestamp. Designed to be called by daily cron.

It also downloads hourly funding rates per pair into the data dir as
<PAIR>_funding_1h.feather (Phase 1 reads these; skip with --no-funding).
With --freqtrade-dir it also writes the files Freqtrade's futures backtester
needs. Freqtrade cannot download Hyperliquid history itself.

Usage:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --pairs BTC/USDC:USDC ETH/USDC:USDC
    uv run python scripts/download_data.py --data-dir /custom/path
    uv run python scripts/download_data.py --freqtrade-dir data/freqtrade
"""

import argparse
import sys
import time
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

# Hyperliquid returns max 500 funding records per request
API_FUNDING_LIMIT = 500

# ccxt multiplies this by the endpoint weight (20 for /info), so 100 ms means
# one request every 2 s, half of Hyperliquid's ~60/min. ccxt's default (1/s)
# gets 429s when a full funding history is pulled.
REQUEST_INTERVAL_MS = 100
MAX_RETRIES = 5


def with_retry(fn, *args, sleep=time.sleep, **kwargs):
    """Call an exchange method, backing off on rate-limit errors (2s, 4s, 8s ...)."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except (ccxt.RateLimitExceeded, ccxt.DDoSProtection):
            if attempt == MAX_RETRIES - 1:
                raise
            sleep(2 ** (attempt + 1))


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
        candles = with_retry(
            exchange.fetch_ohlcv, pair, timeframe, since=current_since, limit=API_CANDLE_LIMIT
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


def funding_filename(pair: str) -> str:
    """Convert pair like 'BTC/USDC:USDC' to 'BTC_USDC-USDC_funding_1h.feather'."""
    return f"{pair.replace('/', '_').replace(':', '-')}_funding_1h.feather"


def download_funding(
    exchange: ccxt.Exchange,
    pair: str,
    data_dir: Path,
    since: str = DEFAULT_SINCE,
) -> int:
    """Download hourly funding rates for a pair. Returns record count."""
    fname = data_dir / funding_filename(pair)
    existing = pd.read_feather(fname) if fname.exists() else pd.DataFrame()
    if not existing.empty:
        since_ms = int(existing["timestamp"].max().timestamp() * 1000) + 1
    else:
        since_ms = exchange.parse8601(f"{since}T00:00:00Z")

    print(f"  Fetching {pair} funding since {datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).isoformat()}")

    rows = []
    while True:
        batch = with_retry(exchange.fetch_funding_rate_history, pair, since=since_ms, limit=API_FUNDING_LIMIT)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < API_FUNDING_LIMIT:
            break
        since_ms = batch[-1]["timestamp"] + 1

    new_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([r["timestamp"] for r in rows], unit="ms", utc=True),
            "rate": [r["fundingRate"] for r in rows],
        }
    )
    if not new_df.empty:
        # Hyperliquid stamps funding a few hundred ms after the hour
        new_df["timestamp"] = new_df["timestamp"].dt.floor("h")
    combined = pd.concat([existing, new_df]) if not existing.empty else new_df
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    combined.to_feather(fname)
    print(f"    {len(combined)} total funding records -> {fname.name}")
    return len(combined)


def freqtrade_pair_name(pair: str) -> str:
    """Convert 'BTC/USDC:USDC' to Freqtrade's file stem 'BTC_USDC_USDC'."""
    return pair.replace("/", "_").replace(":", "_")


def export_freqtrade(pair: str, data_dir: Path, freqtrade_dir: Path) -> None:
    """Write the three files Freqtrade's futures backtester reads for a pair.

    - ``<pair>-4h-futures``: the strategy candles.
    - ``<pair>-1h-futures``: mark candles for funding fees. Hyperliquid only
      serves ~7 months of 1h candles, so these are the 4h candles forward-filled
      to 1h, which covers the full history.
    - ``<pair>-1h-funding_rate``: funding rate in every price column.
    """
    out = freqtrade_dir / "hyperliquid" / "futures"
    out.mkdir(parents=True, exist_ok=True)
    stem = freqtrade_pair_name(pair)

    candles = pd.read_feather(data_dir / pair_to_filename(pair, "4h")).rename(columns={"timestamp": "date"})
    candles["date"] = pd.to_datetime(candles["date"], utc=True)
    candles = candles[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    candles.to_feather(out / f"{stem}-4h-futures.feather")

    mark = candles.set_index("date").resample("1h").ffill().reset_index()
    mark.to_feather(out / f"{stem}-1h-futures.feather")

    funding = pd.read_feather(data_dir / funding_filename(pair))
    fr = pd.DataFrame({"date": funding["timestamp"]})
    for col in ("open", "high", "low", "close"):
        fr[col] = funding["rate"]
    fr["volume"] = 0.0
    fr.reset_index(drop=True).to_feather(out / f"{stem}-1h-funding_rate.feather")
    print(f"    Freqtrade files for {pair} -> {out}")


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
    parser.add_argument(
        "--freqtrade-dir",
        type=Path,
        default=None,
        help="Also write Freqtrade backtest files here (needs funding)",
    )
    parser.add_argument(
        "--no-funding",
        action="store_true",
        help="Skip the funding download (Phase 1 then assumes a constant rate)",
    )
    args = parser.parse_args()
    if args.freqtrade_dir is not None and args.no_funding:
        parser.error("--freqtrade-dir needs funding; drop --no-funding")

    args.data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Hyperliquid OHLCV Download — {datetime.now(timezone.utc).isoformat()}")
    print(f"  Pairs: {args.pairs}")
    print(f"  Timeframes: {args.timeframes}")
    print(f"  Data dir: {args.data_dir}")
    print()

    exchange = ccxt.hyperliquid({
        "options": {"defaultType": "swap"},
        "enableRateLimit": True,
        "rateLimit": REQUEST_INTERVAL_MS,
    })

    errors = []
    for pair in args.pairs:
        for tf in args.timeframes:
            try:
                download_pair(exchange, pair, tf, args.data_dir, args.since)
            except Exception as e:
                msg = f"ERROR downloading {pair} {tf}: {e}"
                print(f"    {msg}", file=sys.stderr)
                errors.append(msg)
        if not args.no_funding:
            try:
                download_funding(exchange, pair, args.data_dir, args.since)
            except Exception as e:
                msg = f"ERROR downloading funding for {pair}: {e}"
                print(f"    {msg}", file=sys.stderr)
                errors.append(msg)
                continue
        if args.freqtrade_dir is not None:
            try:
                export_freqtrade(pair, args.data_dir, args.freqtrade_dir)
            except Exception as e:
                msg = f"ERROR preparing Freqtrade data for {pair}: {e}"
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
