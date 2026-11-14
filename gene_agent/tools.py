"""The two analysis functions from the brief, plus their tool definitions.

Two deliberate changes from the script in the brief (see README, "Changes to
the provided functions"):

1. `get_expressions` takes the cancer indication as well as the gene list.
   The original filtered on gene only, so a gene listed under several cancers
   (TP53 appears under 8) came back with another cancer's value.
2. Cancer names are normalised ("Lung cancer" -> "lung") and an unknown cancer
   raises instead of silently returning an empty result.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "owkin_take_home_data.csv"

df = pd.read_csv(DATA_PATH)

_CANCER_SUFFIX = re.compile(r"\s+(cancers?|carcinomas?|tumou?rs?)$")
# Everyday names that mean exactly one dataset indication.
_ALIASES = {"stomach": "gastric", "kidney": "renal"}
NOT_IN_DATASET = "is not in the dataset"


class UnknownCancerError(ValueError):
    """The requested cancer indication is not in the dataset."""

    def __init__(self, cancer_name: str):
        self.cancer_name = cancer_name
        super().__init__(
            f"'{cancer_name}' {NOT_IN_DATASET}. "
            f"Available cancer indications: {', '.join(available_cancers())}."
        )


def available_cancers() -> List[str]:
    """Every cancer indication present in the dataset, sorted."""
    return sorted(df["cancer_indication"].unique().tolist())


def _plain(cancer_name: str) -> str:
    return _CANCER_SUFFIX.sub("", str(cancer_name).strip().lower())


def normalize_cancer(cancer_name: str) -> str:
    """Map user-style input ("Lung cancer ", "stomach") to the dataset key ("lung", "gastric")."""
    name = _ALIASES.get(_plain(cancer_name), _plain(cancer_name))
    if name not in set(df["cancer_indication"]):
        raise UnknownCancerError(cancer_name)
    return name


def is_cancer_name(text: str) -> bool:
    try:
        normalize_cancer(text)
        return True
    except UnknownCancerError:
        return False


def find_cancer(text: str) -> str:
    """The first dataset cancer (or everyday name for one) mentioned in a text, or ''."""
    return next((word for word in re.findall(r"[a-z]+", text.lower()) if is_cancer_name(word)), "")


def get_targets(cancer_name: str) -> List[str]:
    """Return a list of genes for a given cancer type."""
    cancer = normalize_cancer(cancer_name)
    return df[df["cancer_indication"] == cancer]["gene"].tolist()


def get_expressions(genes: List[str], cancer_name: str) -> Dict[str, float]:
    """Return the median values for the given genes within one cancer type."""
    cancer = normalize_cancer(cancer_name)
    wanted = {str(g).strip().upper() for g in genes}
    subset = df[(df["cancer_indication"] == cancer) & (df["gene"].str.upper().isin(wanted))]
    return dict(zip(subset["gene"].tolist(), subset["median_value"].tolist()))


# --- Tool layer: what the model sees -------------------------------------


def _tool(name: str, description: str, properties: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(properties)},
        },
    }


TOOLS: List[Dict[str, Any]] = [
    _tool(
        "get_targets",
        "List the genes the dataset associates with one cancer. "
        "Call this first for any question about a cancer's genes or their values.",
        {"cancer_name": {"type": "string", "description": "Cancer name, e.g. 'lung' or 'breast'."}},
    ),
    _tool(
        "get_expressions",
        "Get the median expression value of each given gene within one cancer, "
        "plus the median across those genes. Pass the genes returned by get_targets.",
        {
            "genes": {"type": "array", "items": {"type": "string"}, "description": "Gene symbols, e.g. ['KRAS', 'ALK']."},
            "cancer_name": {"type": "string", "description": "The same cancer name used for get_targets."},
        },
    ),
]


def as_gene_list(genes: Any) -> List[str]:
    """Accept a list, or the "KRAS, ALK" string small models sometimes send."""
    if isinstance(genes, str):
        genes = [g for g in re.split(r"[,\s]+", genes.strip("[] ")) if g]
    return [str(g).strip("'\" ") for g in genes or []]


def _asked_as(cancer_name: str) -> Dict[str, str]:
    """Record the everyday name when one was used, so the answer can say so."""
    return {"asked_as": _plain(cancer_name)} if _plain(cancer_name) in _ALIASES else {}


def _run_get_targets(cancer_name: str) -> Dict[str, Any]:
    cancer = normalize_cancer(cancer_name)
    genes = get_targets(cancer)
    return {
        "cancer_indication": cancer,
        **_asked_as(cancer_name),
        "gene_count": len(genes),
        "genes": genes,
        "next_step": f"If the user asked for expression values, now call get_expressions "
        f"with these {len(genes)} genes and cancer_name '{cancer}'.",
    }


def _run_get_expressions(genes: List[str], cancer_name: str) -> Dict[str, Any]:
    genes = as_gene_list(genes)
    cancer = normalize_cancer(cancer_name)
    values = get_expressions(genes, cancer)
    found = {g.upper() for g in values}
    result: Dict[str, Any] = {
        "cancer_indication": cancer,
        **_asked_as(cancer_name),
        "median_values": values,
        "genes_not_found_for_this_cancer": [g for g in genes if g.upper() not in found],
    }
    if values:
        numbers = list(values.values())
        result["summary"] = {
            "gene_count": len(numbers),
            "median_across_genes": round(statistics.median(numbers), 4),
            "min": min(numbers),
            "max": max(numbers),
        }
    return result


_RUNNERS = {"get_targets": _run_get_targets, "get_expressions": _run_get_expressions}


def execute_tool(name: str, tool_input: Dict[str, Any]) -> Tuple[str, bool]:
    """Run a tool call from the model. Returns (content, is_error).

    Errors are returned to the model as text rather than raised, so it can
    explain the problem to the user (e.g. a cancer that is not in the dataset).
    """
    runner = _RUNNERS.get(name)
    if runner is None:
        return f"Unknown tool '{name}'. Available tools: {', '.join(_RUNNERS)}.", True
    try:
        return json.dumps(runner(**tool_input)), False
    except UnknownCancerError as exc:
        return str(exc), True
    except TypeError as exc:  # wrong or missing arguments
        return f"Invalid arguments for {name}: {exc}", True


# Example usage
if __name__ == "__main__":
    cancer = "lung"
    targets = get_targets(cancer)
    expressions = get_expressions(targets, cancer)
    print(f"{cancer} targets:", targets)
    print("Expression values:", expressions)
