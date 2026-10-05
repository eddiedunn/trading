"""python -m agent — write strategies with Claude and backtest them.

    python -m agent run --max-strategies 2 --seed "..."     # print the paper-add command on success
    python -m agent run --queue-paper                       # call paper-add on trinity instead

Needs the `claude` CLI logged in (Eddie's subscription) and a route to the backtest API (it listens on
127.0.0.1:8070 on tela: `ssh -N -L 8070:127.0.0.1:8070 tela`).
"""

import argparse
import logging
import os
import sys

from agent import backtest_client as api
from agent.llm import DEFAULT_MODEL
from agent.loop import Config, run
from agent.runlog import RunLog
from agent.validate import MAX_PARAMS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="write and backtest strategies")
    r.add_argument("--seed", action="append", default=[], help="hypothesis to build on (repeatable)")
    r.add_argument("--max-strategies", type=int, default=2, help="strategies per run (default 2)")
    r.add_argument("--max-iterations", type=int, default=5, help="Claude revisions per strategy (default 5)")
    r.add_argument("--max-phase2", type=int, default=2, help="Phase 2 runs per strategy (default 2; slow)")
    r.add_argument("--max-params", type=int, default=MAX_PARAMS, help="tunable constants allowed per file")
    r.add_argument("--model", default=os.environ.get("AGENT_MODEL", DEFAULT_MODEL))
    r.add_argument("--api-url", default=api.DEFAULT_API_URL, help="backtest API (env BACKTEST_API_URL)")
    r.add_argument("--trinity", default=os.environ.get("TRINITY", "trinity"), help="ssh host for paper-add")
    r.add_argument("--queue-paper", action="store_true",
                   help="call paper-add on trinity for a strategy that passes both phases (never --force)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    if args.cmd == "run":
        cfg = Config(
            seeds=args.seed, max_strategies=args.max_strategies, max_iterations=args.max_iterations,
            max_phase2=args.max_phase2, max_params=args.max_params, model=args.model,
            api_url=args.api_url, queue_paper=args.queue_paper, trinity=args.trinity,
        )
        runlog = RunLog()
        logging.getLogger("agent").info("run %s — log: %s", runlog.run_id, runlog.path)
        outcomes = run(cfg, runlog)
        passed = [o for o in outcomes if o.passed]
        print(f"{len(passed)}/{len(outcomes)} strategies passed both phases. Run log: {runlog.path}")
        return 0 if passed else 1
    return 2
