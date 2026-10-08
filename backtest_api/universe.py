"""The pairs Phase 1 scores. Widened from three coins on 2026-10-08 (Eddie's call).

CORE_PAIRS anchor the scoring periods: ``walk_forward.common_candle_range`` reads their first
and last candles, so a pair listed later (the eight coins have 4h bars from 2024-06-26, the
trade.xyz stock/index/commodity perps from late 2025) contributes the bars it has inside each
period instead of shrinking every period to its own history. ``add_context_columns`` still
attaches only the CONTEXT_COINS closes (BTC, ETH, SOL) to every frame.
"""

CORE_PAIRS = ["BTC_USDC-USDC_4h", "ETH_USDC-USDC_4h", "SOL_USDC-USDC_4h"]
COIN_PAIRS = CORE_PAIRS + [f"{c}_USDC-USDC_4h" for c in ("XRP", "AVAX", "NEAR", "DOGE", "LINK", "UNI", "AAVE", "SUI")]
# Hyperliquid HIP-3 markets deployed by trade.xyz (``xyz:NVDA`` on the API -> ``xyzNVDA`` on disk).
HIP3_PAIRS = [f"xyz{m}_USDC-USDC_4h" for m in ("SP500", "XYZ100", "GOLD", "SILVER", "CL", "NVDA", "MU", "GOOGL", "META", "TSLA", "AMZN", "MSTR")]
# Scored universe (Eddie, 2026-10-08): the coins. The trade.xyz perps stay on disk as data; scoring the
# candidate by its worst commodity (crude: -1.78, 24 trades) judged a crypto-flow rule on a market it never claimed.
PAIRS = COIN_PAIRS
