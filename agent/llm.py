"""Thin wrapper around the Anthropic SDK: one conversation per strategy.

The history is append-only so Claude sees every earlier version and the
feedback it got, and so prompt caching keeps working.
"""

import re

import anthropic

from agent.prompts import initial_prompt, revision_prompt, system_prompt

DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000

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

    def __init__(self, client: anthropic.Anthropic | None = None, model: str = DEFAULT_MODEL):
        # The SDK reads ANTHROPIC_API_KEY itself; nothing here handles the secret.
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.system = system_prompt()
        self.messages: list[dict] = []
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def write(self, seed: str, name: str) -> str:
        return self._ask(initial_prompt(seed, name))

    def revise(self, feedback: str) -> str:
        return self._ask(revision_prompt(feedback))

    def _ask(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            messages=self.messages,
        )
        if response.stop_reason == "refusal":
            raise ReplyError("Claude declined the request")
        if response.stop_reason == "max_tokens":
            raise ReplyError("Claude's reply was cut off at max_tokens")
        self.usage["input_tokens"] += response.usage.input_tokens
        self.usage["output_tokens"] += response.usage.output_tokens
        text = "".join(b.text for b in response.content if b.type == "text")
        # Append the full content so thinking blocks (if any) stay with the turn that produced them.
        self.messages.append({"role": "assistant", "content": response.content})
        return extract_code(text)
