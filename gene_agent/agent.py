"""The agent: a tool-calling loop over the two dataset functions, run on a small local model.

The model decides which function to call and with what arguments; this module
runs the calls, feeds the results back, and stops when the model has an answer.

A small model is good enough to route a question to the right function, but
it can misquote figures. So the facts shown to the user do not depend on its
wording: every reply carries the tool results, rendered by code, and any
number in the model's text that is not in those results is flagged.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ollama import Client, ResponseError

from .tools import TOOLS, available_cancers, execute_tool

DEFAULT_MODEL = "qwen3:1.7b"
# A question needs at most two tool rounds (targets, then expressions); the cap
# only exists to stop a runaway loop.
MAX_STEPS = 6

SYSTEM_PROMPT = """\
You answer questions about a small cancer gene dataset for non-technical users.

The dataset covers only these cancers: {cancers}.

Rules:
- For any question about a cancer's genes, call get_targets with the cancer name.
- For any question about expression values, call get_targets first, then call \
get_expressions with those genes and the same cancer name.
- Use only tool results for gene names and numbers. Never use your own knowledge.
- If a tool says the cancer is not in the dataset, tell the user it is not \
available and list the cancers that are. Give no genes and no numbers.
- If the user uses an everyday name (stomach = gastric, kidney = renal), use the \
dataset name and say so.
- If asked what you can do, say you can list the genes for a cancer and give \
their median expression values, and name the cancers covered. Do not call a tool.
- Answer in one or two short plain sentences. The data table is shown to the \
user separately, so do not repeat every value.\
"""

_DECIMAL = re.compile(r"\d+\.\d+")


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

    def data_markdown(self) -> str:
        """The dataset facts behind this reply, rendered by code, not by the model."""
        final: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for call in self.tool_calls:
            if not call.is_error:
                result = json.loads(call.output)
                final[(call.name, result["cancer_indication"])] = result

        parts = []
        expression_cancers = {cancer for name, cancer in final if name == "get_expressions"}
        for (name, cancer), result in final.items():
            if name == "get_targets" and cancer not in expression_cancers:
                parts.append(f"**Genes for {cancer} ({result['gene_count']}):** {', '.join(result['genes'])}")
            elif name == "get_expressions" and result["median_values"]:
                rows = "\n".join(f"| {gene} | {value} |" for gene, value in result["median_values"].items())
                summary = result["summary"]
                parts.append(
                    f"**Median expression values for {cancer}**\n\n| Gene | Median value |\n|---|---|\n{rows}\n\n"
                    f"Median across these {summary['gene_count']} genes: **{summary['median_across_genes']}**"
                )
        return "\n\n".join(parts)

    def ungrounded_numbers(self) -> List[str]:
        """Decimal numbers in the model's text that appear in no tool result."""
        known = {float(n) for call in self.tool_calls if not call.is_error for n in _DECIMAL.findall(call.output)}
        return [n for n in _DECIMAL.findall(self.text) if float(n) not in known]


class Agent:
    """One conversation. Keeps the message history so follow-ups have context."""

    def __init__(self, client: Optional[Any] = None, model: Optional[str] = None):
        self.client = client or Client()  # reads OLLAMA_HOST, defaults to localhost:11434
        self.model = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL
        self.system = SYSTEM_PROMPT.format(cancers=", ".join(available_cancers()))
        self.messages: List[Any] = []

    def reset(self) -> None:
        self.messages = []

    def ensure_model(self, on_progress: Optional[Callable[[int, int], None]] = None) -> None:
        """Download the model on first use. on_progress receives (done, total) bytes."""
        try:
            self.client.show(self.model)
            return
        except ResponseError as exc:
            if exc.status_code != 404:
                raise
        for part in self.client.pull(self.model, stream=True):
            if on_progress and part.total:
                on_progress(part.completed or 0, part.total)

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
        seen: set = set()
        tools: Optional[List[Dict[str, Any]]] = TOOLS

        for _ in range(MAX_STEPS):
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "system", "content": self.system}, *self.messages],
                tools=tools,
                think=False,
                options={"temperature": 0},
            )
            message = response.message
            self.messages.append(message)

            if not message.tool_calls:
                text = (message.content or "").strip()
                if not text and any(not c.is_error for c in tool_calls):
                    text = "Here is what the dataset contains."
                return Reply(text or "I could not produce an answer. Please try again.", tool_calls), True

            new_calls = 0
            for call in message.tool_calls:
                name, arguments = call.function.name, dict(call.function.arguments or {})
                key = (name, json.dumps(arguments, sort_keys=True))
                if key in seen:
                    output, is_error = "You already made this call. Answer the user now.", True
                else:
                    seen.add(key)
                    new_calls += 1
                    output, is_error = execute_tool(name, arguments)
                    tool_calls.append(ToolCall(name, arguments, output, is_error))
                self.messages.append({"role": "tool", "tool_name": name, "content": output})

            if new_calls == 0:
                # The model is repeating itself: withdraw the tools so the next turn must be an answer.
                tools = None

        return Reply("I could not finish this request within the allowed number of steps.", tool_calls), False
