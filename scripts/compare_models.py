"""Compare local models on the brief's four queries and on paraphrases of them.

    python scripts/compare_models.py qwen3:0.6b lfm3

Needs a running Ollama with the models already pulled. Each check looks at
facts (which data the user was shown, whether the values are the dataset's),
not at wording. The "unseen" questions were written after the loop's guards
were tuned on the first twenty, to see whether the guards generalise.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gene_agent.agent import Agent, Reply  # noqa: E402
from gene_agent.tools import get_expressions, get_targets  # noqa: E402


def shown(reply: Reply) -> set:
    """Cancers whose data the user sees in this reply."""
    return {json.loads(c.output)["cancer_indication"] for c in reply.tool_calls if not c.is_error}


def last_values(reply: Reply) -> dict:
    calls = [json.loads(c.output) for c in reply.tool_calls if c.name == "get_expressions" and not c.is_error]
    return calls[-1]["median_values"] if calls else {}


def no_data(reply: Reply) -> bool:
    return not shown(reply) and reply.written_by in ("model", "fallback")


def genes(cancer: str):
    return lambda r: shown(r) == {cancer} and all(g in r.text for g in get_targets(cancer))


def expression(cancer: str):
    truth = get_expressions(get_targets(cancer), cancer)
    return lambda r: shown(r) == {cancer} and last_values(r) == truth and all(str(v) in r.text for v in truth.values())


def missing(reply: Reply) -> bool:
    return not shown(reply) and (reply.written_by != "model" or "not" in reply.text.lower())


def one_gene(cancer: str, gene: str, value: float):
    return lambda r: shown(r) == {cancer} and last_values(r).get(gene) == value and str(value) in r.text


def only(cancer: str):
    return lambda r: shown(r) == {cancer} and r.written_by == "code"


BRIEF = [
    ("How can you help me?", no_data),
    ("What are the main genes involved in lung cancer?", genes("lung")),
    ("What is the median value expression of genes involved in breast cancer?", expression("breast")),
    ("What is the median value expression of genes involved in esophageal cancer?", missing),
]

TUNED = [
    ("What can you do?", no_data),
    ("Hi, what is this tool for?", no_data),
    ("Which genes are linked to prostate cancer?", genes("prostate")),
    ("List the genes for melanoma", genes("melanoma")),
    ("Tell me the genes associated with ovarian cancer", genes("ovarian")),
    ("What genes are involved in kidney cancer?", genes("renal")),
    ("Which genes are involved in stomach cancer?", genes("gastric")),
    ("Give me the median expression of the genes for lung cancer", expression("lung")),
    ("What are the expression values of pancreatic cancer genes?", expression("pancreatic")),
    ("Show median values for genes in colorectal cancer", expression("colorectal")),
    ("expression of genes in gastric cancer", expression("gastric")),
    ("What is the median expression across renal cancer genes?", expression("renal")),
    ("What are the main genes in liver cancer?", missing),
    ("Median expression of genes in bladder cancer?", missing),
    ("What is the expression value of KRAS in colorectal cancer?", one_gene("colorectal", "KRAS", 0.885)),
    ("What is the median value of TP53 in ovarian cancer?", one_gene("ovarian", "TP53", 0.972)),
]

UNSEEN = [
    ("which genes matter in glioblastoma?", genes("glioblastoma")),
    ("genes for colorectal cancer please", genes("colorectal")),
    ("What genes does the dataset have for breast cancer?", genes("breast")),
    ("pancreatic cancer genes", genes("pancreatic")),
    ("What are the main genes involved in lung cancr?", genes("lung")),
    ("show me expression values for melanoma", expression("melanoma")),
    ("median expression of prostate cancer genes?", expression("prostate")),
    ("How expressed are the genes in ovarian cancer?", expression("ovarian")),
    ("glioblastoma gene expression medians", expression("glioblastoma")),
    ("What about thyroid cancer genes?", missing),
    ("expression values for leukemia", missing),
    ("What does this assistant do?", no_data),
    ("hello", no_data),
    ("Tell me about the weather", no_data),
    ("What is the value of BRAF in melanoma?", one_gene("melanoma", "BRAF", 0.596)),
    ("expression of PTEN in renal cancer", one_gene("renal", "PTEN", 0.399)),
    ("What is the expression of EGFR in lung cancer?", only("lung")),
    (["What are the genes for lung cancer?", "and their expression values?"], expression("lung")),
]


def run(model: str, cases: list) -> tuple:
    """Returns (passed, seconds per question, failed questions)."""
    passed, seconds, failed = 0, [], []
    for question, check in cases:
        agent = Agent(model=model)
        turns = [question] if isinstance(question, str) else question
        start = time.time()
        try:
            for turn in turns:
                reply = agent.ask(turn)
            ok = bool(check(reply))
        except Exception:  # a model that errors out fails the question
            ok = False
        seconds.append(time.time() - start)
        passed += ok
        if not ok:
            failed.append(turns[-1])
    return passed, seconds, failed


if __name__ == "__main__":
    for model in sys.argv[1:]:
        Agent(model=model).ensure_model()
        results = [run(model, cases) for cases in (BRIEF, TUNED, UNSEEN)]
        times = sorted(t for _, seconds, _ in results for t in seconds)
        print(
            f"{model:16} brief {results[0][0]}/{len(BRIEF)}  tuned {results[1][0]}/{len(TUNED)}  "
            f"unseen {results[2][0]}/{len(UNSEEN)}  median {times[len(times) // 2]:.1f}s"
        )
        for _, _, failed in results:
            for question in failed:
                print(f"    failed: {question}")
