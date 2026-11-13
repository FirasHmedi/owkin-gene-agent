"""The agent: a single Claude tool-calling loop over the two dataset functions.

The model decides which function to call and with what arguments; this module
runs the calls, feeds the results back, and stops when the model has an answer.
Every number in an answer comes from a tool result, and each reply carries the
list of tool calls behind it so the UI can show how an answer was produced.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import anthropic

from .tools import TOOLS, available_cancers, execute_tool

DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000
# A question needs at most two tool rounds (targets, then expressions); the cap
# only exists to stop a runaway loop.
MAX_STEPS = 8

SYSTEM_PROMPT = """\
You help non-technical stakeholders explore a small oncology dataset by asking \
questions in plain language. The dataset links cancer indications to the genes \
associated with them, and gives one median expression value per gene per cancer.

Cancer indications in the dataset: {cancers}.

You have two tools: get_targets (genes for a cancer) and get_expressions \
(median values for genes within a cancer). To answer a question about expression \
values for a cancer, call get_targets first and pass its genes to get_expressions \
with the same cancer.

Every gene list and every number you give must come from a tool result in this \
conversation. People will act on these figures as "what our dataset says", so do \
not fill gaps from general biomedical knowledge. If a cancer is not in the dataset, \
say so plainly and list the indications that are available; do not substitute a \
related cancer or textbook genes.

When the user uses an everyday name for something in the dataset (for example \
"stomach" for gastric, "kidney" for renal), use the dataset name and mention which \
one you used.

The dataset has no ranking of genes, so for questions about the "main" genes of a \
cancer, give all the genes listed for it and say that this is the full list in the \
dataset. For expression questions, give the value per gene and the median across \
those genes from the tool's summary; do not calculate statistics yourself. The \
values have no unit recorded, so do not attach one.

Write for a reader with no bioinformatics background: short, direct answers, a \
table when there are several values, no jargon without a brief explanation. When \
asked what you can do, describe these capabilities and offer two or three example \
questions. You describe data; you do not give medical advice.\
"""


@dataclass
class ToolCall:
    name: str
    input: Dict[str, Any]
    output: str
    is_error: bool


@dataclass
class Reply:
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)


class Agent:
    """One conversation. Keeps the message history so follow-ups have context."""

    def __init__(self, client: Optional[Any] = None, model: Optional[str] = None):
        self.client = client or anthropic.Anthropic()
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
        self.system = SYSTEM_PROMPT.format(cancers=", ".join(available_cancers()))
        self.messages: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.messages = []

    def ask(self, question: str) -> Reply:
        """Answer one user question, running tools as the model requests them."""
        checkpoint = len(self.messages)
        self.messages.append({"role": "user", "content": question})
        completed = False
        try:
            reply, completed = self._run()
        finally:
            if not completed:
                # Drop the unfinished exchange so the next question starts clean.
                del self.messages[checkpoint:]
        return reply

    def _run(self) -> Tuple[Reply, bool]:
        """Returns the reply and whether the exchange completed normally."""
        tool_calls: List[ToolCall] = []

        for _ in range(MAX_STEPS):
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=self.system,
                tools=TOOLS,
                messages=self.messages,
                output_config={"effort": "low"},
                # If a safety classifier declines the request (biomedical wording
                # can trigger false positives), retry on Anthropic's recommended
                # fallback model inside the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )

            if response.stop_reason == "refusal":
                return Reply("The model declined to answer this request. Please try rephrasing it.", tool_calls), False

            # Keep the full content (thinking and tool_use blocks included): the
            # API needs it unchanged on the next request.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "pause_turn":
                continue

            if response.stop_reason != "tool_use":
                text = "\n\n".join(b.text for b in response.content if b.type == "text").strip()
                if response.stop_reason == "max_tokens":
                    text += "\n\n(The answer was cut off because it reached the length limit.)"
                return Reply(text or "I could not produce an answer. Please try again.", tool_calls), True

            # All results for one assistant turn go back in a single user message.
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                output, is_error = execute_tool(block.name, dict(block.input))
                tool_calls.append(ToolCall(block.name, dict(block.input), output, is_error))
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error}
                )
            self.messages.append({"role": "user", "content": results})

        return Reply("I could not finish this request within the allowed number of steps.", tool_calls), False
