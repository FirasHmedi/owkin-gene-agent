"""Turn tool results into the text the user reads.

Answers that contain data are written here, by code, so every gene and number
on screen comes straight from a tool result. The model's own wording is used
only for replies without data, and `invented_terms` is the check applied to it.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .tools import NOT_IN_DATASET, available_cancers

if TYPE_CHECKING:
    from .agent import ToolCall

_DECIMAL = re.compile(r"\d+\.\d+")
_SYMBOL = re.compile(r"\b[A-Z][A-Z0-9]{2,7}\b")  # looks like a gene symbol: TP53, KRAS
_NOT_GENES = {"DNA", "RNA", "CSV", "NOT", "AND", "THE", "YOU", "CAN"}  # capitalised words that are not genes


def capabilities() -> str:
    """One line on what the assistant can do, added to every reply that has no data."""
    return (
        "I can list the genes the dataset links to a cancer, and give their median expression values. "
        f"Cancers covered: {', '.join(available_cancers())}."
    )


def invented_terms(text: str) -> List[str]:
    """Numbers and gene-like symbols in a model reply that had no tool result behind it.

    A reply without data should contain neither, so anything found here means
    the model answered from memory.
    """
    return _DECIMAL.findall(text) + [s for s in _SYMBOL.findall(text) if s not in _NOT_GENES]


def write_answer(tool_calls: List["ToolCall"], question: str = "") -> Optional[str]:
    """Write the answer from the tool results of one question.

    Returns None when the tools produced nothing to report, in which case the
    caller falls back to the model's own reply.
    """
    # Keep the latest successful result per cancer, for each of the two tools.
    targets: Dict[str, Dict[str, Any]] = {}
    expressions: Dict[str, Dict[str, Any]] = {}
    for call in tool_calls:
        if not call.is_error:
            result = json.loads(call.output)
            by_tool = targets if call.name == "get_targets" else expressions
            by_tool[result["cancer_indication"]] = result

    # One paragraph per cancer, in the order the cancers were first looked up.
    # Expression values include the gene names, so they replace the plain gene list.
    paragraphs = []
    for cancer in dict.fromkeys([*targets, *expressions]):
        if cancer in expressions:
            paragraphs.append(_expressions_text(cancer, expressions[cancer], question))
        else:
            paragraphs.append(_targets_text(cancer, targets[cancer]))
    if paragraphs:
        return "\n\n".join(paragraphs)

    # No data at all: if a lookup failed because the cancer is unknown, say so.
    for call in tool_calls:
        if call.is_error and NOT_IN_DATASET in call.output:
            return call.output
    return None


def _label(cancer: str) -> str:
    """ "lung" -> "lung cancer", but "melanoma" stays "melanoma"."""
    return cancer if cancer in {"glioblastoma", "melanoma"} else f"{cancer} cancer"


def _alias_note(result: Dict[str, Any]) -> str:
    """Tell the user when their everyday name was mapped to a dataset name."""
    asked_as = result.get("asked_as")
    return f'"{asked_as.capitalize()}" is listed as {result["cancer_indication"]} in the dataset. ' if asked_as else ""


def _targets_text(cancer: str, result: Dict[str, Any]) -> str:
    genes = result["genes"]
    return f"{_alias_note(result)}The dataset lists {len(genes)} genes for {_label(cancer)}: {', '.join(genes)}."


def _expressions_text(cancer: str, result: Dict[str, Any], question: str = "") -> str:
    values: Dict[str, float] = result["median_values"]
    not_found: List[str] = result["genes_not_found_for_this_cancer"]
    note = _alias_note(result)

    if not values:
        return f"{note}The dataset has no value for {', '.join(not_found)} in {_label(cancer)}."

    if len(values) == 1:
        gene, value = next(iter(values.items()))
        text = f"{note}In {_label(cancer)}, {gene} has a median expression value of **{value}**."
    else:
        # "Median value expression of genes" can mean each gene's value or one
        # median across them, so the answer gives the table and the overall median.
        summary = result["summary"]
        rows = "\n".join(f"| {gene} | {value} |" for gene, value in values.items())
        # If the user asked about particular genes but the model fetched them all,
        # answer the question first and keep the full table as context.
        asked = [g for g in values if re.search(rf"\b{re.escape(g)}\b", question, re.IGNORECASE)]
        for gene in asked if len(asked) < len(values) else []:
            note += f"In {_label(cancer)}, {gene} has a median expression value of **{values[gene]}**. "
        text = (
            f"{note}Median expression values for {len(values)} genes in {_label(cancer)}:\n\n"
            f"| Gene | Median value |\n|---|---|\n{rows}\n\n"
            f"The median across these {summary['gene_count']} genes is **{summary['median_across_genes']}** "
            f"(lowest {summary['min']}, highest {summary['max']})."
        )

    if not_found:
        text += f"\n\nNot listed for {_label(cancer)}: {', '.join(not_found)}."
    return text
