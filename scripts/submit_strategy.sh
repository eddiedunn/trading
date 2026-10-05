#!/usr/bin/env bash
# Run a strategy file through Phase 1, Phase 2 and the final test on tela, then
# queue it for the paper arena on trinity. Stops at the first phase that fails.
#
# The final test (phase 3) runs on the held-back data and is allowed once per
# strategy (and per exact code) per campaign: the API refuses a second one.
# Read its result before trusting the strategy.
#
# Usage:
#   scripts/submit_strategy.sh strategies/examples/EmaCross.py           # stop if a phase fails
#   scripts/submit_strategy.sh strategies/examples/EmaCross.py --force   # queue anyway (pipeline test)
#
# With --force a failing Phase 1 or Phase 2 doesn't stop the script, but the
# final test is still skipped unless Phase 2 passed (the API would refuse it).
#
# The file must define generate_signals(df) for Phase 1 and an IStrategy class
# named after the file for Phase 2, the final test and paper (see
# strategies/examples/EmaCross.py).

set -euo pipefail

file="${1:?usage: submit_strategy.sh <Strategy.py> [--force]}"
force="${2:-}"
name="$(basename "$file" .py)"
TRINITY="${TRINITY:-trinity}"
TELA="${TELA:-tela}"
API="http://127.0.0.1:8070"
PAPER_DIR="/data/services/trading/paper"
out="$(mktemp -d)"

request() {  # phase
  python3 -c 'import json,sys; print(json.dumps({"strategy_name": sys.argv[1], "phase": int(sys.argv[2]), "strategy_code": open(sys.argv[3]).read()}))' \
    "$name" "$1" "$file"
}

passed() {  # json file
  python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("passed") else 1)' "$1"
}

run_phase() {  # phase, label, output file; returns 1 if the API refused or the phase failed
  echo "== $2: $name"
  local resp code
  resp="$(request "$1" | ssh "$TELA" "curl -sS -m 1800 -X POST $API/backtest -H 'Content-Type: application/json' -d @- -w '\n%{http_code}'")"
  code="${resp##*$'\n'}"
  printf '%s\n' "${resp%$'\n'*}" > "$3"
  if [ "$code" != "200" ]; then
    echo "$2 refused or failed (HTTP $code):"
    cat "$3"
    return 1
  fi
  python3 -m json.tool "$3"
  if ! passed "$3"; then
    echo "$2 did not pass."
    return 1
  fi
}

stop_unless_forced() {
  if [ "$force" != "--force" ]; then
    echo "Stopping."
    exit 1
  fi
}

run_phase 1 "Phase 1" "$out/phase1.json" || stop_unless_forced
p2_ok=yes
run_phase 2 "Phase 2" "$out/phase2.json" || { p2_ok=no; stop_unless_forced; }

final_args=""
if [ "$p2_ok" = yes ]; then
  run_phase 3 "Final test (held-back data, once per strategy)" "$out/final_test.json" || stop_unless_forced
  [ -s "$out/final_test.json" ] && final_args="--final-test $PAPER_DIR/final_test.json"
else
  echo "== Final test skipped: Phase 2 did not pass"
fi

echo "== Queue for paper arena"
ssh "$TRINITY" "mkdir -p /tmp/submit-$name"
scp -q "$file" "$out"/*.json "$TRINITY:/tmp/submit-$name/"
ssh "$TRINITY" "cp /tmp/submit-$name/* $PAPER_DIR/ && rm -rf /tmp/submit-$name && \
  podman exec paper-arena-monitor python -m live.trading_client paper-add \
    --strategy $name --file $PAPER_DIR/$name.py \
    --phase1 $PAPER_DIR/phase1.json --phase2 $PAPER_DIR/phase2.json $final_args $force; \
  rc=\$?; rm -f $PAPER_DIR/$name.py $PAPER_DIR/phase1.json $PAPER_DIR/phase2.json $PAPER_DIR/final_test.json; exit \$rc"
