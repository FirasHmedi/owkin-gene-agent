"""The agent: a tool-calling loop over the two dataset functions, run on a very small local model.

A model of a few hundred megabytes can work out what the user is asking and
pick the right function. It cannot be trusted to plan two steps reliably or to
quote figures, so the loop keeps those jobs in code:

1. The model chooses which tools to call (and may chain them).
2. The loop checks the arguments: genes must come from the user or the dataset.
3. Any answer that contains data is written by code from the tool results.
4. The model's own wording is used only for replies without data, and is
   rejected if it mentions a gene or a number.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ollama import Client, ResponseError

from .tools import (
    NOT_IN_DATASET,
    TOOLS,
    as_gene_list,
    available_cancers,
    execute_tool,
    find_cancer,
    is_cancer_name,
)

DEFAULT_MODEL = "qwen3:0.6b"
# A question needs at most two tool rounds (targets, then expressions); the cap
# only exists to stop a runaway loop.
MAX_STEPS = 5
MAX_REPLY_TOKENS = 300

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

_DECIMAL = re.compile(r"\d+\.\d+")
_SYMBOL = re.compile(r"\b[A-Z][A-Z0-9]{2,7}\b")  # looks like a gene symbol: TP53, KRAS
_NOT_GENES = {"DNA", "RNA", "CSV", "NOT", "AND", "THE", "YOU", "CAN"}


@dataclass
class ToolCall:
    name: str
    input: Dict[str, Any]
    output: str
    is_error: bool
    added_by_agent: bool = False  # run by the loop itself, not requested by the model


@dataclass
class Reply:
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    written_by: str = "code"  # "code" (from tool results), "model", or "fallback" (model reply rejected)


def capabilities() -> str:
    return (
        "I can list the genes the dataset links to a cancer, and give their median expression values. "
        f"Cancers covered: {', '.join(available_cancers())}."
    )


def _label(cancer: str) -> str:
    return cancer if cancer in {"glioblastoma", "melanoma"} else f"{cancer} cancer"


def write_answer(tool_calls: List[ToolCall]) -> Optional[str]:
    """Write the answer from tool results. None if the tools gave nothing to report."""
    targets: Dict[str, Dict[str, Any]] = {}
    expressions: Dict[str, Dict[str, Any]] = {}
    for call in tool_calls:
        if not call.is_error:
            result = json.loads(call.output)
            (targets if call.name == "get_targets" else expressions)[result["cancer_indication"]] = result

    parts = []
    for cancer in dict.fromkeys([*targets, *expressions]):  # each cancer once, in order of first lookup
        alias = (targets.get(cancer) or expressions[cancer]).get("asked_as")
        prefix = f'"{alias.capitalize()}" is listed as {cancer} in the dataset. ' if alias else ""
        if cancer not in expressions:
            genes = targets[cancer]["genes"]
            parts.append(f"{prefix}The dataset lists {len(genes)} genes for {_label(cancer)}: {', '.join(genes)}.")
            continue

        result = expressions[cancer]
        values, summary = result["median_values"], result.get("summary")
        missing = result["genes_not_found_for_this_cancer"]
        if not values:
            parts.append(f"{prefix}The dataset has no value for {', '.join(missing)} in {_label(cancer)}.")
            continue
        if len(values) == 1:
            (gene, value), = values.items()
            text = f"{prefix}In {_label(cancer)}, {gene} has a median expression value of **{value}**."
        else:
            rows = "\n".join(f"| {gene} | {value} |" for gene, value in values.items())
            text = (
                f"{prefix}Median expression values for {len(values)} genes in {_label(cancer)}:\n\n"
                f"| Gene | Median value |\n|---|---|\n{rows}\n\n"
                f"The median across these {summary['gene_count']} genes is **{summary['median_across_genes']}** "
                f"(lowest {summary['min']}, highest {summary['max']})."
            )
        if missing:
            text += f"\n\nNot listed for {_label(cancer)}: {', '.join(missing)}."
        parts.append(text)
    if parts:
        return "\n\n".join(parts)

    for call in tool_calls:
        if call.is_error and NOT_IN_DATASET in call.output:
            return call.output
    return None


def invented_terms(text: str) -> List[str]:
    """Numbers and gene-like symbols in a reply that had no tool result to take them from."""
    return _DECIMAL.findall(text) + [s for s in _SYMBOL.findall(text) if s not in _NOT_GENES]


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
        try:
            model_text, tool_calls = self._run_tools()
        except Exception:
            del self.messages[checkpoint:]  # drop the unfinished exchange
            raise

        answer = write_answer(tool_calls)
        if answer:
            reply = Reply(answer, tool_calls, "code")
        elif model_text and not invented_terms(model_text):
            reply = Reply(f"{model_text}\n\n{capabilities()}", tool_calls, "model")
        else:
            reply = Reply(f"I could not match that to the dataset. {capabilities()}", tool_calls, "fallback")

        # Keep what the user actually saw as the assistant turn, so follow-ups build on it.
        del self.messages[checkpoint + 1 :]
        self.messages.append({"role": "assistant", "content": reply.text})
        return reply

    def _run_tools(self) -> Tuple[str, List[ToolCall]]:
        """Let the model call tools until it stops. Returns its final text and the calls made."""
        tool_calls: List[ToolCall] = []
        seen: set = set()

        for _ in range(MAX_STEPS):
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "system", "content": self.system}, *self.messages],
                tools=TOOLS,
                think=False,
                options={"temperature": 0, "num_predict": MAX_REPLY_TOKENS},
            )
            message = response.message
            self.messages.append(message)
            if not message.tool_calls:
                return (message.content or "").strip(), tool_calls

            new_calls = 0
            for call in message.tool_calls:
                name, arguments = call.function.name, dict(call.function.arguments or {})
                key = (name, json.dumps(arguments, sort_keys=True))
                if key in seen:
                    output = "You already made this call."
                else:
                    seen.add(key)
                    new_calls += 1
                    if name == "get_expressions":
                        arguments = self._ground_genes(arguments, tool_calls)
                    output, is_error = execute_tool(name, arguments)
                    tool_calls.append(ToolCall(name, arguments, output, is_error))
                self.messages.append({"role": "tool", "tool_name": name, "content": output})

            if new_calls == 0:  # the model is repeating itself
                break
        return "", tool_calls

    def _ground_genes(self, arguments: Dict[str, Any], tool_calls: List[ToolCall]) -> Dict[str, Any]:
        """Make sure the genes passed to get_expressions come from the user or the dataset.

        Small models tend to skip get_targets and pass genes from memory, or put
        the cancer name in the gene list. If any gene appears neither in the
        conversation nor in an earlier tool result, the loop runs get_targets
        itself and uses the dataset's list.
        """
        genes = as_gene_list(arguments.get("genes"))
        said = [str(m["content"]) for m in self.messages if isinstance(m, dict) and m["role"] in ("user", "tool")]
        question = next(m["content"] for m in reversed(self.messages) if isinstance(m, dict) and m["role"] == "user")
        # A missing cancer name is taken from the gene list or the question, never guessed.
        cancer = arguments.get("cancer_name") or next((g for g in genes if is_cancer_name(g)), find_cancer(question))
        arguments = {**arguments, "cancer_name": cancer}

        sources = " ".join(said).upper()
        grounded = [g for g in genes if not is_cancer_name(g) and re.search(rf"\b{re.escape(g.upper())}\b", sources)]
        if genes and len(grounded) == len(genes):
            return arguments

        lookup = {"cancer_name": cancer}
        output, is_error = execute_tool("get_targets", lookup)
        if is_error:
            return arguments  # get_expressions will report the same problem
        tool_calls.append(ToolCall("get_targets", lookup, output, False, added_by_agent=True))
        return {**arguments, "genes": json.loads(output)["genes"]}
