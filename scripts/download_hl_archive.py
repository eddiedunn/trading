#!/usr/bin/env python3
"""
Hyperliquid archive downloader: per-minute asset contexts (open interest, premium,
mark/oracle/mid price, impact prices, daily volume) from the requester-pays S3 bucket
``hyperliquid-archive``, resampled to hourly per pair.

Writes ``<PAIR>_ctx_1h.feather`` into the data dir, the same naming as the candle and
funding files (``BTC_USDC-USDC_ctx_1h.feather``). Raw daily files are cached under
``<data-dir>/hl_archive/`` (gitignored) so reruns only fetch new days.

Needs: an AWS login (``aws login`` or an access key with s3:GetObject on the bucket),
the ``aws`` CLI and the ``lz4`` binary on PATH. Egress is billed to the requester,
about 7 MB per day of history (under $1 for the whole archive).

Known traps (github.com/alpenmilch411/hyperliquid-archive-notes): zeros mean "absent",
not zero; open interest is in coin units, not USD; some days are missing from the
bucket; a few rows carry duplicate minutes. This script maps zero to NaN, keeps OI in
coin units, tolerates missing days and takes the last row per minute.

Usage:
    uv run python scripts/download_hl_archive.py                 # since 2023-11-01, default pairs
    uv run python scripts/download_hl_archive.py --since 2025-01-01 --coins BTC ETH
"""

import argparse
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

BUCKET = "s3://hyperliquid-archive/asset_ctxs"
DEFAULT_COINS = ["BTC", "ETH", "SOL"]
DEFAULT_SINCE = "2023-11-01"
# archive columns we keep, and their names in the output
KEEP = {
    "open_interest": "open_interest",
    "premium": "premium",
    "mark_px": "mark_px",
    "oracle_px": "oracle_px",
    "mid_px": "mid_px",
    "impact_bid_px": "impact_bid_px",
    "impact_ask_px": "impact_ask_px",
    "day_ntl_vlm": "day_ntl_vlm",
    "funding": "funding_archive",
}


def pair_filename(coin: str) -> str:
    return f"{coin}_USDC-USDC_ctx_1h.feather"


def fetch_day(day: date, raw_dir: Path) -> Path | None:
    """Download one day's lz4 CSV if not cached. Returns the path, or None if the bucket lacks the day."""
    out = raw_dir / f"{day:%Y%m%d}.csv.lz4"
    if out.exists() and out.stat().st_size > 0:
        return out
    cmd = ["aws", "s3", "cp", "--request-payer", "requester", f"{BUCKET}/{day:%Y%m%d}.csv.lz4", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        if "404" in r.stderr or "Not Found" in r.stderr or "NoSuchKey" in r.stderr:
            return None
        raise RuntimeError(f"aws s3 cp failed for {day}: {r.stderr.strip()[:300]}")
    return out


def read_day(path: Path, coins: list[str]) -> pd.DataFrame:
    """Parse one archive day into minute rows for the chosen coins."""
    raw = subprocess.run(["lz4", "-dc", str(path)], capture_output=True, check=True).stdout.decode()
    df = pd.read_csv(StringIO(raw))
    if "time" not in df.columns or "coin" not in df.columns:
        raise RuntimeError(f"unexpected archive header: {list(df.columns)}")
    df = df[df["coin"].isin(coins)].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates(["coin", "time"], keep="last")
    return df


def to_hourly(minutes: pd.DataFrame) -> pd.DataFrame:
    """Hourly bars per coin: last snapshot of each hour for levels, mean for premium.

    Zero is the archive's encoding for "absent", so it becomes NaN before aggregating.
    """
    cols = [c for c in KEEP if c in minutes.columns]
    m = minutes.set_index("time")[cols].astype(float).replace(0.0, np.nan)
    agg = {c: "last" for c in cols}
    agg["premium"] = "mean"
    hourly = m.resample("1h", label="left", closed="left").agg(agg)
    hourly = hourly.rename(columns=KEEP)
    hourly.index.name = "timestamp"
    return hourly.reset_index()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    ap.add_argument("--since", default=DEFAULT_SINCE)
    ap.add_argument("--until", default=None, help="last day inclusive (default: yesterday UTC)")
    ap.add_argument("--coins", nargs="+", default=DEFAULT_COINS)
    ap.add_argument("--no-fetch", action="store_true", help="only rebuild the feathers from cached days")
    args = ap.parse_args(argv)

    raw_dir = args.data_dir / "hl_archive"
    raw_dir.mkdir(parents=True, exist_ok=True)
    start = date.fromisoformat(args.since)
    end = date.fromisoformat(args.until) if args.until else datetime.now(timezone.utc).date() - timedelta(days=1)

    frames, missing, fetched = [], [], 0
    day = start
    while day <= end:
        path = raw_dir / f"{day:%Y%m%d}.csv.lz4"
        if not path.exists() and not args.no_fetch:
            got = fetch_day(day, raw_dir)
            if got is None:
                missing.append(day)
                day += timedelta(days=1)
                continue
            fetched += 1
        if path.exists():
            frames.append(read_day(path, args.coins))
        else:
            missing.append(day)
        day += timedelta(days=1)
    print(f"days fetched now: {fetched}, days missing from the bucket: {len(missing)}"
          + (f" (first few: {[d.isoformat() for d in missing[:5]]})" if missing else ""))
    if not frames:
        print("no data", file=sys.stderr)
        return 1
    minutes = pd.concat(frames, ignore_index=True)
    for coin in args.coins:
        h = to_hourly(minutes[minutes["coin"] == coin])
        oi = h["open_interest"].dropna()
        if oi.empty:
            print(f"{coin}: no rows in the archive for these days, nothing written")
            continue
        out = args.data_dir / pair_filename(coin)
        h.to_feather(out)
        print(f"{coin}: {len(h)} hourly rows {h['timestamp'].min():%Y-%m-%d} .. {h['timestamp'].max():%Y-%m-%d}, "
              f"open interest {oi.iloc[0]:,.0f} -> {oi.iloc[-1]:,.0f} {coin}, {h['open_interest'].isna().mean()*100:.1f}% hours missing -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
