"""Thin wrapper around the `claude` CLI: one conversation per strategy.

Runs `claude -p` so calls go through Eddie's Claude subscription login, not an
API key. The first call starts a session; each revision resumes it, so Claude
sees every earlier version and the feedback it got.
"""

import json
import os
import re
import subprocess
import tempfile

from agent.prompts import initial_prompt, revision_prompt, system_prompt

DEFAULT_MODEL = "claude-opus-5-5"
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
TIMEOUT_SECS = 600

_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


class ReplyError(Exception):
    """Claude's reply had no usable code block."""


def extract_code(text: str) -> str:
    blocks = _FENCE_RE.findall(text)
    if not blocks:
        raise ReplyError("Reply did not contain a ```python code block")
    return blocks[-1].strip() + "\n"


class StrategyWriter:
    """Drives one Claude conversation that writes and then revises a single strategy."""

    def __init__(self, model: str = DEFAULT_MODEL, runner=subprocess.run):
        self.model = model
        self.runner = runner
        self.system = system_prompt()
        self.session_id: str | None = None
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    def write(self, seed: str, name: str) -> str:
        return self._ask(initial_prompt(seed, name))

    def revise(self, feedback: str) -> str:
        return self._ask(revision_prompt(feedback))

    def _ask(self, user_text: str) -> str:
        cmd = [
            CLAUDE_BIN, "-p", "--output-format", "json", "--model", self.model,
            "--system-prompt", self.system,
            # Text only: no tools, no user/project settings, hooks, MCP servers or skills.
            "--tools", "", "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands",
        ]
        if self.session_id:
            cmd += ["--resume", self.session_id]
        # Drop any API key so the CLI uses the subscription login.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        proc = self.runner(cmd, input=user_text, capture_output=True, text=True,
                           timeout=TIMEOUT_SECS, env=env, cwd=tempfile.gettempdir())
        try:
            reply = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise ReplyError(f"claude exited {proc.returncode}: {(proc.stderr or proc.stdout)[-500:]}")
        if reply.get("is_error"):
            raise ReplyError(f"claude reported an error: {reply.get('result') or reply.get('subtype')}")
        if reply.get("stop_reason") == "refusal":
            raise ReplyError("Claude declined the request")
        if reply.get("stop_reason") == "max_tokens":
            raise ReplyError("Claude's reply was cut off at max_tokens")
        self.session_id = reply["session_id"]
        usage = reply.get("usage") or {}
        self.usage["input_tokens"] += usage.get("input_tokens", 0)
        self.usage["output_tokens"] += usage.get("output_tokens", 0)
        self.usage["cost_usd"] += reply.get("total_cost_usd") or 0.0
        return extract_code(reply.get("result") or "")
