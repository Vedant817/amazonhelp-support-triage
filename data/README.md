# Data

## What is committed

| File | Size | Role |
|---|---:|---|
| `processed/amazonhelp_context_pairs.csv` | 17 MB | The corpus. Every entry point reads this: the API, the model factory, the score matrix and all diagnostics. |
| `training/labels_for_human_review.csv` | 0.2 MB | The held-out evaluation set used by `--holdout review`. An active-learning sample, i.e. the hardest rows in the corpus. |

## What is not committed, and why

| File | Why |
|---|---|
| `raw/twcs.csv` | The Twitter customer-support corpus exceeds 50 MB, so `.gitignore` excludes it. |
| `processed/amazonhelp_pairs.csv`, `.jsonl` | Intermediates of the rebuild below. Regenerable, so not tracked. |

## Rebuilding the corpus

The committed corpus is an artifact of record: the pipeline reads it directly and
nothing needs to be rebuilt to run the service. To regenerate it from raw data:

```bash
# 1. Download the Twitter customer-support dataset (twcs.csv) and place it here:
#    data/raw/twcs.csv
# 2. Rebuild:
python -m src.data_pipeline
```

`src/data_pipeline.py` runs two passes over the raw file:

1. `process_amazon_conversations` extracts customer/agent pairs and writes
   `processed/amazonhelp_pairs.csv`.
2. `add_prior_conversation_context` follows raw reply links backwards to
   reconstruct earlier turns, and writes the context-enriched corpus.

The second pass exists because the model factory requires `conversation_context`
and `prior_turn_count`. Reconstruction is deliberately leak-safe: the current
customer message and any later support reply are never placed in the context, so
training and evaluation see only what a live system would have observed at
decision time.

Useful flags: `--raw`, `--output`, `--max-pairs`, `--max-prior-turns`.

## Caveats on the labels

The intent targets are **weak supervision**, extracted from historical agent
replies by rule — not human judgement. They encode what the agent did, while the
model only ever sees what the customer wrote, so part of every label is
unlearnable by construction. This is the dominant term in the remaining error;
see the README's "On supervision quality" section. `scripts_label_workflow.py`
is the path to replacing these targets with human labels.
