#!/usr/bin/env bash
# Run a strategy file through Phase 1 and Phase 2 on tela, then queue it for
# the paper arena on trinity if both passed.
#
# Usage:
#   scripts/submit_strategy.sh strategies/examples/EmaCross.py           # stop if a phase fails
#   scripts/submit_strategy.sh strategies/examples/EmaCross.py --force   # queue anyway (pipeline test)
#
# The file must define generate_signals(df) for Phase 1 and an IStrategy class
# named after the file for Phases 2-4 (see strategies/examples/EmaCross.py).

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

for phase in 1 2; do
  echo "== Phase $phase: $name"
  request "$phase" | ssh "$TELA" "curl -sf -m 1800 -X POST $API/backtest -H 'Content-Type: application/json' -d @-" > "$out/phase$phase.json"
  python3 -m json.tool "$out/phase$phase.json"
done

echo "== Queue for paper arena"
ssh "$TRINITY" "mkdir -p /tmp/submit-$name"
scp -q "$file" "$out/phase1.json" "$out/phase2.json" "$TRINITY:/tmp/submit-$name/"
ssh "$TRINITY" "cp /tmp/submit-$name/* $PAPER_DIR/ && rm -rf /tmp/submit-$name && \
  podman exec paper-arena-monitor python -m live.trading_client paper-add \
    --strategy $name --file $PAPER_DIR/$name.py \
    --phase1 $PAPER_DIR/phase1.json --phase2 $PAPER_DIR/phase2.json $force; \
  rc=\$?; rm -f $PAPER_DIR/$name.py $PAPER_DIR/phase1.json $PAPER_DIR/phase2.json; exit \$rc"
