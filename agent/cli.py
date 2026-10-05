"""python -m agent — write strategies with Claude and backtest them.

    python -m agent run --max-strategies 2 --seed "..."

A strategy that passes Phase 1 and Phase 2 is saved and the agent prints the
scripts/submit_strategy.sh command for you to run the final test on the
held-back data; the agent never runs the final test or queues for paper.

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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    if args.cmd == "run":
        cfg = Config(
            seeds=args.seed, max_strategies=args.max_strategies, max_iterations=args.max_iterations,
            max_phase2=args.max_phase2, max_params=args.max_params, model=args.model,
            api_url=args.api_url,
        )
        runlog = RunLog()
        logging.getLogger("agent").info("run %s — log: %s", runlog.run_id, runlog.path)
        outcomes = run(cfg, runlog)
        passed = [o for o in outcomes if o.passed]
        print(f"{len(passed)}/{len(outcomes)} strategies passed both phases and await a human's final test. "
              f"Run log: {runlog.path}")
        return 0 if passed else 1
    return 2
