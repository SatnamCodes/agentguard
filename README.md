# AgentGuard

AgentGuard prepares red-team conversation traces for safety-focused analysis
and experimentation.

## Overview

The data-preparation scripts download Anthropic's `hh-rlhf` red-team-attempts
subset, extract each conversation transcript and its harmlessness score, derive
a binary label, and save the processed records as JSON.

## Repository structure

```text
src/data/dataset.py              Dataset generation script (20,000 records)
src/data/loader.py               Smaller generation variant (2,000 records)
src/data/processed_traces.json   Generated dataset output
notebooks/data_gen.ipynb         Exploratory data-generation notebook
requirements.txt                 Python dependencies
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate data

Run from the repository root:

```bash
python src/data/dataset.py
```

The command requires network access to Hugging Face on its first run and writes
the processed data to `src/data/processed_traces.json`.

## Data record format

```json
{
  "text": "conversation transcript",
  "label": 0,
  "score": 0.0
}
```

`text` is limited to 2,000 characters. `score` is the source harmlessness
score, and `label` is `1` when the score is negative and `0` otherwise.
