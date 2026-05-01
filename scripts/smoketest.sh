#!/usr/bin/env bash
# End-to-end smoke test for the deployed trading stack.
# Validates: backtest API health, postgres reachability, paper monitor + live
# bot service status, trading_client status, and a Phase 1 backtest round-trip.
#
# Usage:
#   scripts/smoketest.sh                   # uses defaults below
#   BACKTEST_URL=... TRINITY=... scripts/smoketest.sh
#
# Exit code: 0 = all green, non-zero = first failure.

set -euo pipefail

BACKTEST_URL="${BACKTEST_URL:-https://backtest.starbluesolutions.net}"
TRINITY="${TRINITY:-trinity}"
TELA="${TELA:-tela}"

pass() { printf "  \033[32m✓\033[0m %s\n" "$*"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$*"; exit 1; }
step() { printf "\n\033[1m== %s ==\033[0m\n" "$*"; }

step "1/5 backtest API health"
code=$(curl -s -o /dev/null -w '%{http_code}' "${BACKTEST_URL}/health" || true)
[[ "$code" == "200" ]] && pass "GET ${BACKTEST_URL}/health → 200" || fail "got HTTP $code"

step "2/5 trinity systemd units active"
for unit in trading-postgres paper-arena-monitor trading-live; do
  state=$(ssh "$TRINITY" "systemctl --user is-active ${unit}.service" 2>/dev/null || true)
  [[ "$state" == "active" ]] && pass "${unit}.service: $state" || fail "${unit}.service: ${state:-unknown}"
done

step "3/5 postgres tables present"
tables=$(ssh "$TRINITY" "podman exec trading-postgres psql -U trading -d trading -tAc \
  \"SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='public' \
   AND tablename IN ('paper_snapshots','strategy_registry') ORDER BY tablename\"" 2>/dev/null)
expected=$'paper_snapshots\nstrategy_registry'
[[ "$tables" == "$expected" ]] && pass "paper_snapshots, strategy_registry present" || fail "tables: $tables"

step "4/5 trading_client status reachable"
out=$(ssh "$TRINITY" "podman exec trading-live python -m live.trading_client status 2>&1" || true)
echo "$out" | grep -q "active_strategy.txt" && pass "status reports active strategy" || fail "$out"

step "5/5 Phase 1 backtest round-trip"
code=$(curl -s -o /tmp/smoketest_backtest.json -w '%{http_code}' \
  -X POST "${BACKTEST_URL}/backtest" \
  -H 'Content-Type: application/json' \
  -d '{
    "strategy_name": "_smoketest_null",
    "phase": 1,
    "strategy_code": "import pandas as pd\ndef generate_signals(df):\n    s = pd.Series(0, index=df.index)\n    return s\n"
  }' || true)
[[ "$code" == "200" ]] && pass "POST /backtest phase=1 → 200" || fail "got HTTP $code (body: $(cat /tmp/smoketest_backtest.json))"

printf "\n\033[1;32mAll smoke tests passed.\033[0m\n"
