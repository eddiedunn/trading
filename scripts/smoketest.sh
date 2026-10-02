#!/usr/bin/env bash
# End-to-end smoke test for the deployed trading stack.
# Validates: backtest API health + a Phase 1 round-trip on tela, postgres
# tables, the paper monitor and live bot services, and that the live bot is in
# dry-run.
#
# Usage:
#   scripts/smoketest.sh                   # uses defaults below
#   TRINITY=... TELA=... scripts/smoketest.sh
#
# Exit code: 0 = all green, non-zero = first failure.

set -euo pipefail

TRINITY="${TRINITY:-trinity}"
TELA="${TELA:-tela}"
API="http://127.0.0.1:8070"  # reached through ssh to tela

pass() { printf "  \033[32m✓\033[0m %s\n" "$*"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$*"; exit 1; }
step() { printf "\n\033[1m== %s ==\033[0m\n" "$*"; }

step "1/5 backtest API health (tela)"
code=$(ssh "$TELA" "curl -s -o /dev/null -w '%{http_code}' $API/health" || true)
[[ "$code" == "200" ]] && pass "GET /health → 200" || fail "got HTTP $code"

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

step "4/5 live bot answers and is in dry-run"
ping=$(ssh "$TRINITY" "curl -s -m5 http://127.0.0.1:8080/api/v1/ping" || true)
[[ "$ping" == *pong* ]] && pass "/api/v1/ping → pong" || fail "ping: $ping"
mode=$(ssh "$TRINITY" "grep -o '\"dry_run\": *[a-z]*' /opt/trading/live/config.json" || true)
[[ "$mode" == *true* ]] && pass "config: $mode" || fail "live config is not dry-run: $mode"

step "5/5 Phase 1 backtest round-trip"
body='{"strategy_name": "SmoketestNull", "phase": 1, "strategy_code": "import pandas as pd\ndef generate_signals(df):\n    return pd.Series(0, index=df.index)\n"}'
out=$(ssh "$TELA" "curl -s -w '\n%{http_code}' -X POST $API/backtest -H 'Content-Type: application/json' -d @-" <<<"$body" || true)
code=$(tail -n1 <<<"$out")
[[ "$code" == "200" ]] && pass "POST /backtest phase=1 → 200" || fail "got HTTP $code (body: $(head -n1 <<<"$out"))"

printf "\n\033[1;32mAll smoke tests passed.\033[0m\n"
