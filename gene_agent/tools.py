"""The two analysis functions from the brief.

Two deliberate changes from the script in the brief:

1. `get_expressions` takes the cancer indication as well as the gene list.
   The original filtered on gene only, so a gene listed under several cancers
   (TP53 appears under 8) came back with another cancer's value.
2. Cancer names are normalised ("Lung cancer" -> "lung") and an unknown cancer
   raises instead of silently returning an empty result.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "owkin_take_home_data.csv"

df = pd.read_csv(DATA_PATH)

_CANCER_SUFFIX = re.compile(r"\s+(cancers?|carcinomas?|tumou?rs?)$")


class UnknownCancerError(ValueError):
    """The requested cancer indication is not in the dataset."""

    def __init__(self, cancer_name: str):
        self.cancer_name = cancer_name
        super().__init__(
            f"'{cancer_name}' is not in the dataset. "
            f"Available cancer indications: {', '.join(available_cancers())}."
        )


def available_cancers() -> List[str]:
    """Every cancer indication present in the dataset, sorted."""
    return sorted(df["cancer_indication"].unique().tolist())


def normalize_cancer(cancer_name: str) -> str:
    """Map user-style input ("Lung cancer ") to the dataset key ("lung")."""
    name = _CANCER_SUFFIX.sub("", str(cancer_name).strip().lower())
    if name not in set(df["cancer_indication"]):
        raise UnknownCancerError(cancer_name)
    return name


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


# Example usage
if __name__ == "__main__":
    cancer = "lung"
    targets = get_targets(cancer)
    expressions = get_expressions(targets, cancer)
    print(f"{cancer} targets:", targets)
    print("Expression values:", expressions)
