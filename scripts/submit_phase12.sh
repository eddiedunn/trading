#!/usr/bin/env bash
# Run a strategy through Phase 1 and Phase 2 on tela only. Does NOT run the
# final test or queue paper: use scripts/submit_strategy.sh for that once the
# Phase 2 result has been read. Results are saved next to the strategy file as
# <Name>.phase1.json / <Name>.phase2.json.
set -euo pipefail
file="${1:?usage: submit_phase12.sh <Strategy.py>}"
name="$(basename "$file" .py)"
TELA="${TELA:-tela}"
API="http://127.0.0.1:8070"
dir="$(dirname "$file")"

request() {  # phase
  python3 -c 'import json,sys; print(json.dumps({"strategy_name": sys.argv[1], "phase": int(sys.argv[2]), "strategy_code": open(sys.argv[3]).read()}))' \
    "$name" "$1" "$file"
}

run_phase() {  # phase, label, output file
  echo "== $2: $name"
  local resp code
  resp="$(request "$1" | ssh "$TELA" "curl -sS -m 1800 -X POST $API/backtest -H 'Content-Type: application/json' -d @- -w '\n%{http_code}'")"
  code="${resp##*$'\n'}"
  printf '%s\n' "${resp%$'\n'*}" > "$3"
  python3 -m json.tool "$3" || cat "$3"
  if [ "$code" != "200" ]; then echo "$2 refused or failed (HTTP $code)"; return 1; fi
  python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("passed") else 1)' "$3" || { echo "$2 did not pass."; return 1; }
}

run_phase 1 "Phase 1" "$dir/$name.phase1.json"
run_phase 2 "Phase 2" "$dir/$name.phase2.json"
echo "== Phase 1 and Phase 2 passed. Final test NOT run. Next: scripts/submit_strategy.sh $file"
