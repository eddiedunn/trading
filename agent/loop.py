"""The agent loop: write, validate, Phase 1, Phase 2, revise on failure.

One strategy at a time; each gets its own Claude conversation. A strategy that
passes both phases is written to the run folder with its API responses, and
the command to queue it for the paper arena is printed. Nothing here can
promote: the only trinity-side call is ``paper-add`` (never ``--force``), and
only with ``--queue-paper``.
"""

import logging
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from agent import backtest_client as api
from agent.llm import DEFAULT_MODEL, ReplyError, StrategyWriter
from agent.prompts import DEFAULT_SEEDS
from agent.runlog import RunLog
from agent.validate import MAX_PARAMS, validate_strategy

log = logging.getLogger("agent")

TRINITY_PAPER_DIR = "/data/services/trading/paper"


@dataclass
class Config:
    seeds: list[str] = field(default_factory=list)
    max_strategies: int = 2
    max_iterations: int = 5  # Claude revisions per strategy, across validation, Phase 1 and Phase 2
    max_phase2: int = 2  # Phase 2 runs per strategy (minutes each)
    max_params: int = MAX_PARAMS
    model: str = DEFAULT_MODEL
    api_url: str = api.DEFAULT_API_URL
    queue_paper: bool = False
    trinity: str = "trinity"
    name_prefix: str = "Agent"


@dataclass
class Outcome:
    name: str
    passed: bool
    attempts: int
    phase2_runs: int
    reason: str
    paths: dict[str, Path] = field(default_factory=dict)


def run(cfg: Config, runlog: RunLog, writer_factory=None) -> list[Outcome]:
    writer_factory = writer_factory or (lambda: StrategyWriter(model=cfg.model))
    seeds = cfg.seeds or DEFAULT_SEEDS
    outcomes = []
    for i in range(cfg.max_strategies):
        seed = seeds[i % len(seeds)]
        # Unique per run so candidates never overwrite each other on tela, e.g. Agent2510051412N1.
        stamp = "".join(ch for ch in runlog.run_id if ch.isdigit())[2:12]
        name = f"{cfg.name_prefix}{stamp}N{i + 1}"
        log.info("== Strategy %d/%d: %s", i + 1, cfg.max_strategies, name)
        log.info("   seed: %s", seed)
        outcome = _develop_one(cfg, runlog, writer_factory(), name, seed)
        outcomes.append(outcome)
        runlog.record(name, outcome.attempts, "result", outcome.passed, reason=outcome.reason,
                      phase2_runs=outcome.phase2_runs, paths={k: str(v) for k, v in outcome.paths.items()})
        log.info("   %s: %s", "PASSED" if outcome.passed else "stopped", outcome.reason)
    return outcomes


def _develop_one(cfg: Config, runlog: RunLog, writer: StrategyWriter, name: str, seed: str) -> Outcome:
    attempts = 0
    phase2_runs = 0
    feedback = None
    phase1_result = None

    while attempts < cfg.max_iterations:
        attempts += 1
        try:
            code = writer.write(seed, name) if feedback is None else writer.revise(feedback)
        except ReplyError as e:
            runlog.record(name, attempts, "llm", False, error=str(e))
            return Outcome(name, False, attempts, phase2_runs, f"Claude reply unusable: {e}")
        runlog.save_code(name, attempts, code)
        runlog.record(name, attempts, "llm", True, usage=dict(writer.usage))

        problems = validate_strategy(code, name, cfg.max_params)
        runlog.record(name, attempts, "validate", not problems, problems=problems)
        if problems:
            log.info("   v%d: validation failed: %s", attempts, "; ".join(problems))
            feedback = "The file failed local validation:\n- " + "\n- ".join(problems)
            continue

        try:
            phase1_result = api.run_phase(name, code, 1, cfg.api_url)
        except api.BacktestError as e:
            runlog.record(name, attempts, "phase1", False, error=str(e))
            log.info("   v%d: phase 1 error", attempts)
            feedback = str(e)
            continue
        runlog.record(name, attempts, "phase1", phase1_result.get("passed"), stats=phase1_result.get("stats"))
        log.info("   v%d: phase 1 %s", attempts, "passed" if phase1_result.get("passed") else "failed")
        if not phase1_result.get("passed"):
            feedback = api.phase1_feedback(phase1_result)
            continue

        if phase2_runs >= cfg.max_phase2:
            return Outcome(name, False, attempts, phase2_runs, "Phase 2 budget used up")
        phase2_runs += 1
        try:
            phase2_result = api.run_phase(name, code, 2, cfg.api_url)
        except api.BacktestError as e:
            runlog.record(name, attempts, "phase2", False, error=str(e))
            log.info("   v%d: phase 2 error", attempts)
            feedback = str(e)
            continue
        runlog.record(name, attempts, "phase2", phase2_result.get("passed"), windows=phase2_result.get("windows"))
        log.info("   v%d: phase 2 %s", attempts, "passed" if phase2_result.get("passed") else "failed")
        if not phase2_result.get("passed"):
            feedback = api.phase1_feedback(phase1_result) + "\n" + api.phase2_feedback(phase2_result)
            continue

        paths = runlog.save_final(name, code, phase1_result, phase2_result)
        _hand_off(cfg, name, paths)
        return Outcome(name, True, attempts, phase2_runs, "passed Phase 1 and Phase 2", paths)

    return Outcome(name, False, attempts, phase2_runs, "iteration budget used up")


def paper_add_commands(name: str, paths: dict[str, Path], trinity: str = "trinity") -> list[str]:
    """The exact commands a human runs to queue the strategy for the paper arena."""
    code, p1, p2 = paths["code"], paths["phase1"], paths["phase2"]
    remote = f"{TRINITY_PAPER_DIR}/{name}"
    return [
        f"scripts/submit_strategy.sh {shlex.quote(str(code))}",
        # Or, reusing the saved results instead of re-running both phases:
        f"scp {shlex.quote(str(code))} {shlex.quote(str(p1))} {shlex.quote(str(p2))} {trinity}:{TRINITY_PAPER_DIR}/ && "
        f"ssh {trinity} podman exec paper-arena-monitor python -m live.trading_client paper-add "
        f"--strategy {name} --file {remote}.py --phase1 {TRINITY_PAPER_DIR}/{p1.name} --phase2 {TRINITY_PAPER_DIR}/{p2.name}",
    ]


def queue_paper(name: str, paths: dict[str, Path], trinity: str = "trinity") -> None:
    """Copy the file and results to trinity and run paper-add there. Never passes --force."""
    code, p1, p2 = paths["code"], paths["phase1"], paths["phase2"]
    subprocess.run(["scp", "-q", str(code), str(p1), str(p2), f"{trinity}:{TRINITY_PAPER_DIR}/"], check=True)
    remote_code = f"{TRINITY_PAPER_DIR}/{name}.py"
    remote_p1 = f"{TRINITY_PAPER_DIR}/{p1.name}"
    remote_p2 = f"{TRINITY_PAPER_DIR}/{p2.name}"
    try:
        subprocess.run(
            ["ssh", trinity, "podman", "exec", "paper-arena-monitor", "python", "-m", "live.trading_client",
             "paper-add", "--strategy", name, "--file", remote_code, "--phase1", remote_p1, "--phase2", remote_p2],
            check=True,
        )
    finally:
        subprocess.run(["ssh", trinity, "rm", "-f", remote_code, remote_p1, remote_p2], check=False)


def _hand_off(cfg: Config, name: str, paths: dict[str, Path]) -> None:
    if cfg.queue_paper:
        log.info("   queueing %s for the paper arena on %s", name, cfg.trinity)
        queue_paper(name, paths, cfg.trinity)
        return
    print(f"\n{name} passed both phases. Saved to {paths['code']}. To queue it for the paper arena, run one of:")
    for cmd in paper_add_commands(name, paths, cfg.trinity):
        print(f"  {cmd}")
    print()
