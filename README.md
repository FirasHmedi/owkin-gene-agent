# Gene data assistant

A proof-of-concept agent that lets non-technical users query a cancer gene dataset in plain language.

Work in progress. This first step contains the data layer only: the dataset and the two analysis functions the agent will orchestrate.

## Run it

Requires Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m gene_agent.tools
```

This prints the genes and median expression values for lung cancer.

## Changes to the provided functions

- `get_expressions(genes, cancer_name)` takes the cancer as well as the genes. The original looked up by gene only, so a gene listed under several cancers (TP53 appears under 8) returned another cancer's value.
- Cancer names are normalised ("Lung cancer" becomes "lung"), and an unknown cancer raises an error listing the available ones.
