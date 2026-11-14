"""The agent: a tool-calling loop over the two dataset functions, run on a small local model.

The model reads the question and decides which function to call. A small model
does that well, but it can skip a step or misquote a figure, so the loop does
not rely on it for facts:

1. The model chooses which tools to call, and can chain them.
2. Before `get_expressions` runs, the loop checks that its genes come from the
   user or from the dataset, and fetches the dataset's list if they do not.
3. Any answer that contains data is written by code from the tool results.
4. The model's own wording is used only for replies without data, and is
   dropped if it mentions a gene or a number.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ollama import Client, ResponseError

from .answers import capabilities, invented_terms, write_answer
from .tools import TOOLS, as_gene_list, available_cancers, execute_tool, find_cancer, is_cancer_name

DEFAULT_MODEL = "qwen3:0.6b"  # override with the OLLAMA_MODEL environment variable

# A question needs at most two tool rounds (targets, then expressions). The cap
# only exists to stop a model that keeps calling tools.
MAX_STEPS = 5
MAX_REPLY_TOKENS = 300  # stops a model that starts listing things endlessly

SYSTEM_PROMPT = """\
You route questions about a cancer gene dataset to tools.

Cancers in the dataset: {cancers}.

- If the user asks which genes are involved in a cancer, call get_targets.
- If the user asks for expression values, call get_targets, then call \
get_expressions with those genes and the same cancer.
- Always call a tool when the question names a cancer, even one that is not in \
the list. The tool reports missing cancers.
- Never answer about genes or values from memory.
- If the question is not about the data (for example "how can you help me?"), \
reply in one or two sentences.\
"""


@dataclass
class ToolCall:
    """One call to a dataset function, kept so the UI can show how an answer was produced."""

    name: str
    input: Dict[str, Any]
    output: str  # JSON on success, a plain message on error
    is_error: bool
    added_by_agent: bool = False  # run by the loop itself, not requested by the model


@dataclass
class Reply:
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    # Who wrote `text`: "code" (from tool results), "model" (no data involved),
    # or "fallback" (the model's reply was dropped by the invented-terms check).
    written_by: str = "code"


class Agent:
    """One conversation. Keeps the message history so follow-up questions have context."""

    def __init__(self, client: Optional[Any] = None, model: Optional[str] = None):
        self.client = client or Client()  # reads OLLAMA_HOST, defaults to localhost:11434
        self.model = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL
        self.system = SYSTEM_PROMPT.format(cancers=", ".join(available_cancers()))
        self.messages: List[Any] = []

    def reset(self) -> None:
        self.messages = []

    def ensure_model(self, on_progress: Optional[Callable[[int, int], None]] = None) -> None:
        """Download the model if the server does not have it yet.

        on_progress, if given, is called with (bytes done, bytes total).
        """
        try:
            self.client.show(self.model)
            return
        except ResponseError as exc:
            if exc.status_code != 404:  # 404 means "not downloaded yet"
                raise
        for part in self.client.pull(self.model, stream=True):
            if on_progress and part.total:
                on_progress(part.completed or 0, part.total)

    def ask(self, question: str) -> Reply:
        """Answer one question."""
        start = len(self.messages)
        self.messages.append({"role": "user", "content": question})
        try:
            model_text, tool_calls = self._run_tools()
        except Exception:
            del self.messages[start:]  # leave the history as it was before this question
            raise

        answer = write_answer(tool_calls)
        if answer:
            reply = Reply(answer, tool_calls, "code")
        elif model_text and not invented_terms(model_text):
            reply = Reply(f"{model_text}\n\n{capabilities()}", tool_calls, "model")
        else:
            reply = Reply(f"I could not match that to the dataset. {capabilities()}", tool_calls, "fallback")

        # Replace the raw tool exchange with what the user actually saw, so a
        # follow-up ("and their expression values?") builds on the shown answer.
        del self.messages[start + 1 :]
        self.messages.append({"role": "assistant", "content": reply.text})
        return reply

    def _run_tools(self) -> Tuple[str, List[ToolCall]]:
        """Let the model call tools until it stops asking.

        Returns the model's final text and the list of calls that were run.
        """
        tool_calls: List[ToolCall] = []
        already_run = set()

        for _ in range(MAX_STEPS):
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "system", "content": self.system}, *self.messages],
                tools=TOOLS,
                think=False,  # no hidden reasoning pass: routing does not need it
                options={"temperature": 0, "num_predict": MAX_REPLY_TOKENS},
            )
            message = response.message
            self.messages.append(message)
            if not message.tool_calls:
                return (message.content or "").strip(), tool_calls

            ran_something_new = False
            for call in message.tool_calls:
                name, arguments = call.function.name, dict(call.function.arguments or {})
                key = (name, json.dumps(arguments, sort_keys=True))
                if key in already_run:
                    output = "You already made this call."
                else:
                    already_run.add(key)
                    ran_something_new = True
                    if name == "get_expressions":
                        arguments = self._check_expression_arguments(arguments, tool_calls)
                    output, is_error = execute_tool(name, arguments)
                    tool_calls.append(ToolCall(name, arguments, output, is_error))
                self.messages.append({"role": "tool", "tool_name": name, "content": output})

            if not ran_something_new:  # the model is repeating itself
                break
        return "", tool_calls

    def _check_expression_arguments(self, arguments: Dict[str, Any], tool_calls: List[ToolCall]) -> Dict[str, Any]:
        """Make sure a get_expressions call only uses facts the model was given.

        Small models sometimes skip get_targets and pass genes from memory, put
        the cancer name in the gene list, or leave the cancer out. This returns
        arguments in which the cancer comes from the call or the question, and
        every gene comes from the conversation or from the dataset.
        """
        genes = as_gene_list(arguments.get("genes"))
        conversation = [m for m in self.messages if isinstance(m, dict)]  # user and tool messages
        question = next(m["content"] for m in reversed(conversation) if m["role"] == "user")

        # The cancer: as given, else a cancer name found in the gene list, else one in the question.
        cancer = arguments.get("cancer_name") or next((g for g in genes if is_cancer_name(g)), find_cancer(question))
        arguments = {**arguments, "cancer_name": cancer}

        # The genes: accepted only if each one was typed by the user or returned by a tool.
        given = " ".join(str(m["content"]) for m in conversation).upper()
        known = [g for g in genes if not is_cancer_name(g) and re.search(rf"\b{re.escape(g.upper())}\b", given)]
        if genes and len(known) == len(genes):
            return arguments

        # Otherwise do the step the model skipped: ask the dataset for this cancer's genes.
        lookup = {"cancer_name": cancer}
        output, is_error = execute_tool("get_targets", lookup)
        if is_error:
            return arguments  # unknown cancer: get_expressions will report it
        tool_calls.append(ToolCall("get_targets", lookup, output, False, added_by_agent=True))
        return {**arguments, "genes": json.loads(output)["genes"]}
