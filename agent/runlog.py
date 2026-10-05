"""Per-run folder under agent_runs/ (gitignored): a JSON-lines log of every
attempt plus each version of each strategy file, so a run can be read back
without the API responses."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.environ.get("AGENT_RUNS_DIR", str(_REPO_ROOT / "agent_runs")))


class RunLog:
    def __init__(self, runs_dir: Path = RUNS_DIR, run_id: str | None = None):
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = runs_dir / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "log.jsonl"

    def record(self, strategy: str, attempt: int, stage: str, passed: bool | None, **fields) -> dict:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "strategy": strategy,
            "attempt": attempt,
            "stage": stage,  # llm | validate | phase1 | phase2 | result
            "passed": passed,
            **fields,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry

    def save_code(self, strategy: str, attempt: int, code: str) -> Path:
        path = self.dir / f"{strategy}.v{attempt}.py"
        path.write_text(code)
        return path

    def save_final(self, strategy: str, code: str, phase1: dict, phase2: dict) -> dict[str, Path]:
        """The passing version plus the two API responses paper-add wants."""
        paths = {
            "code": self.dir / f"{strategy}.py",
            "phase1": self.dir / f"{strategy}.phase1.json",
            "phase2": self.dir / f"{strategy}.phase2.json",
        }
        paths["code"].write_text(code)
        paths["phase1"].write_text(json.dumps(phase1, indent=2))
        paths["phase2"].write_text(json.dumps(phase2, indent=2))
        return paths
