"""Unit tests for the Freqtrade export and funding download in scripts/download_data.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import ccxt  # noqa: E402
import pytest  # noqa: E402

from download_data import (  # noqa: E402
    MAX_RETRIES,
    with_retry,
    download_funding,
    export_freqtrade,
    freqtrade_pair_name,
    funding_filename,
    pair_to_filename,
)

PAIR = "BTC/USDC:USDC"


def _write_inputs(data_dir: Path):
    ts = pd.date_range("2025-01-01", periods=3, freq="4h", tz="UTC")
    pd.DataFrame(
        {"timestamp": ts, "open": [1.0, 2, 3], "high": [1.0, 2, 3], "low": [1.0, 2, 3],
         "close": [1.0, 2, 3], "volume": [10.0, 20, 30]}
    ).to_feather(data_dir / pair_to_filename(PAIR, "4h"))
    fts = pd.date_range("2025-01-01", periods=9, freq="1h", tz="UTC")
    pd.DataFrame({"timestamp": fts, "rate": [0.0001] * 9}).to_feather(data_dir / funding_filename(PAIR))


def test_freqtrade_pair_name():
    assert freqtrade_pair_name(PAIR) == "BTC_USDC_USDC"


def test_export_writes_candles_mark_and_funding(tmp_path):
    _write_inputs(tmp_path)
    ft = tmp_path / "ft"

    export_freqtrade(PAIR, tmp_path, ft)

    out = ft / "hyperliquid" / "futures"
    candles = pd.read_feather(out / "BTC_USDC_USDC-4h-futures.feather")
    assert list(candles.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(candles) == 3

    mark = pd.read_feather(out / "BTC_USDC_USDC-1h-futures.feather")
    assert len(mark) == 9  # 00:00 .. 08:00 hourly
    assert mark["close"].tolist()[:5] == [1.0, 1.0, 1.0, 1.0, 2.0]

    funding = pd.read_feather(out / "BTC_USDC_USDC-1h-funding_rate.feather")
    assert (funding["open"] == 0.0001).all()
    assert (funding["volume"] == 0).all()


def test_download_funding_floors_to_hour_and_appends(tmp_path):
    exchange = MagicMock()
    exchange.parse8601.return_value = 0
    first_ms = int(pd.Timestamp("2025-01-01T00:00:00.351Z").timestamp() * 1000)
    exchange.fetch_funding_rate_history.return_value = [
        {"timestamp": first_ms, "fundingRate": 1e-5},
        {"timestamp": first_ms + 3_600_000, "fundingRate": 2e-5},
    ]

    assert download_funding(exchange, PAIR, tmp_path) == 2
    df = pd.read_feather(tmp_path / funding_filename(PAIR))
    assert df["timestamp"].iloc[0] == pd.Timestamp("2025-01-01T00:00:00Z")

    # Second run resumes after the last record and dedupes overlap
    exchange.fetch_funding_rate_history.return_value = [
        {"timestamp": first_ms + 3_600_000, "fundingRate": 2e-5},
        {"timestamp": first_ms + 7_200_000, "fundingRate": 3e-5},
    ]
    assert download_funding(exchange, PAIR, tmp_path) == 3
    since = exchange.fetch_funding_rate_history.call_args.kwargs["since"]
    assert since > int(pd.Timestamp("2025-01-01T01:00:00Z").timestamp() * 1000)


def test_with_retry_backs_off_on_rate_limit_then_succeeds():
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ccxt.RateLimitExceeded("429")
        return "ok"

    assert with_retry(flaky, sleep=sleeps.append) == "ok"
    assert sleeps == [2, 4]


def test_with_retry_gives_up_after_max_retries():
    def always_limited():
        raise ccxt.RateLimitExceeded("429")

    sleeps = []
    with pytest.raises(ccxt.RateLimitExceeded):
        with_retry(always_limited, sleep=sleeps.append)
    assert len(sleeps) == MAX_RETRIES - 1


def _run_main(monkeypatch, tmp_path, *extra):
    import download_data

    calls = {"funding": [], "export": []}
    monkeypatch.setattr(download_data.ccxt, "hyperliquid", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr(download_data, "download_pair", lambda *a, **k: 0)
    monkeypatch.setattr(download_data, "download_funding",
                        lambda ex, pair, data_dir, since: calls["funding"].append((pair, data_dir)))
    monkeypatch.setattr(download_data, "export_freqtrade",
                        lambda pair, data_dir, ft: calls["export"].append(pair))
    monkeypatch.setattr(sys, "argv", ["download_data.py", "--pairs", PAIR, "--data-dir", str(tmp_path), *extra])
    download_data.main()
    return calls


def test_main_downloads_funding_into_data_dir_by_default(monkeypatch, tmp_path):
    """Phase 1 reads funding from the data dir, so it is fetched even without --freqtrade-dir."""
    calls = _run_main(monkeypatch, tmp_path)
    assert calls["funding"] == [(PAIR, tmp_path)]
    assert calls["export"] == []


def test_main_no_funding_skips_it(monkeypatch, tmp_path):
    calls = _run_main(monkeypatch, tmp_path, "--no-funding")
    assert calls["funding"] == []


def test_main_freqtrade_export_still_runs(monkeypatch, tmp_path):
    calls = _run_main(monkeypatch, tmp_path, "--freqtrade-dir", str(tmp_path / "ft"))
    assert calls["funding"] == [(PAIR, tmp_path)]
    assert calls["export"] == [PAIR]
