# @AmazonHelp support-agent assignment

This repository implements a local, evidence-first support pipeline over historical
`@AmazonHelp` conversations. It does not call a hosted model API.

The current system is:

1. non-vetoable safety gates for public PII, account takeover, and legal threats;
2. one revision-pinned MiniLM encoder with separate intent and escalation logistic heads;
3. temperature scaling on reply-labeled holdouts;
4. confidence abstention and a recall-targeted escalation threshold;
5. BM25 + MiniLM retrieval fused with reciprocal-rank fusion;
6. a resolution act extracted from historical replies;
7. either local Qwen rewriting or deterministic act-conditioned offline rewriting;
8. URL, PII, false-promise, and 280-character guardrails.

## Verified status

`results/headline_results.json` is the only headline source. The saved artifact was
produced with `python main.py benchmark --include-zero-shot-local`; the plain benchmark
command omits that optional CPU-heavy baseline. On the audited 200-case gold set:

| System | Intent macro-F1 | Escalate recall | Escalate precision | Reason exact match |
|---|---:|---:|---:|---:|
| Majority + always escalate | 0.061 | 1.000 | 0.550 | 0.000 |
| TF-IDF + keyword escalation + retrieve/replay | 0.673 | 0.309 | 0.630 | 0.000 |
| Zero-shot local Qwen | 0.355 | 0.018 | 1.000 | 0.210 |
| MiniLM dual-head + gates + retrieve/rewrite | **0.731** | **0.873** | 0.558 | **0.455** |

The proposed system's bootstrap 95% intervals are:

- intent macro-F1: 0.665–0.790;
- escalation recall: 0.806–0.932;
- escalation precision: 0.485–0.634.

These results are materially safer than the keyword baseline on escalation recall, but
they do **not** meet the intended intent-quality or escalation-precision targets. There
are currently zero independently human-reviewed training labels. The prepared 400-row
review sheet is `data/training/labels_for_human_review.csv`; blank review fields are not
treated as labels.

The full zero-shot local baseline produced 14 invalid structured outputs and averaged
4.95 seconds per case in this CPU-only environment. No LLM-judge score or judge–human agreement is claimed in the default run. Those fields
are explicitly `not_run` / `not_available` unless the local judge runs and matching
human scores exist for the same reply IDs.

## Install and run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py test
.\.venv\Scripts\python.exe main.py benchmark
.\.venv\Scripts\python.exe main.py serve
```

The first learned run downloads the pinned
`sentence-transformers/all-MiniLM-L6-v2` revision. Model files use the normal Hugging
Face cache and are not committed.

The API defaults to deterministic offline generation so it can be reproduced without
loading a generative model. To use the pinned local Qwen model:

```powershell
$env:SUPPORT_GENERATION_MODE = "local"
.\.venv\Scripts\python.exe main.py serve
```

Run the local generator, zero-shot local baseline, and local judge explicitly:

```powershell
.\.venv\Scripts\python.exe main.py benchmark `
  --generation-mode local `
  --include-zero-shot-local `
  --judge-local
```

This is substantially slower on CPU. The local-model benchmark does not invent a human
agreement score. First have reviewers complete the generated
`results/human_rating_template.csv`, save matching ratings at
`data/evaluation/human_same_reply_ratings.csv`, and rerun.

## API

- `GET /api/v1/health` reports actual model temperatures, threshold, corpus size, and generation mode.
- `POST /api/v1/confidence` returns calibrated intent probabilities and the abstention decision.
- `POST /api/v1/triage` returns intent, learned escalation risk, exact reason code, resolution act, PII-stripped retrieval evidence, guarded draft, and latency.
- `GET /metrics` exposes operational counters. These are observability, not quality evidence.

## Label and leakage controls

- The gold tweet IDs are removed before supervision and retrieval are built.
- Intent weak labels are inferred from AmazonHelp's reply, not from the customer's keywords.
- Whether Amazon moved a conversation to DM/phone/chat supplies the escalation weak label.
- Ambiguous historical replies remain unlabeled.
- `scripts/audit_gold_labels.py` applies only four documented policy corrections; see
  `data/golden/LABEL_AUDIT.md`.
- Independent reviewers can override weak labels by filling the prepared 400-row sheet.

## Evaluation contract

Headline metrics are macro-F1, escalation recall/precision, exact reason-code match,
bootstrap intervals, and reply safety invariants. The benchmark also writes:

- `results/golden_predictions.csv` — row-level evidence;
- `results/failure_examples.csv` — the first ten actual gold errors;
- `results/human_rating_template.csv` — same system replies for independent rating;
- `results/baseline_comparison.csv` — concise comparison table.

ROUGE/BLEU are not treated as reply-quality metrics. Conformal risk guarantees, energy
OOD, CRAG, and a keyword-based “LLM judge” are not part of the scored production path.

## Method scope

MiniLM maps text into 384-dimensional vectors suitable for semantic similarity
([model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)). This
implementation currently trains heads over frozen embeddings; it is not claimed as
SetFit fine-tuning. SetFit remains the planned method after independent labels exist
([Tunstall et al., 2022](https://arxiv.org/abs/2209.11055)). Temperature scaling follows
the single-parameter calibration method of
[Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html).

The retrieve-then-rewrite design is inspired by retrieval-augmented generation, but is
described by what it actually does rather than claimed as the exact RAG training method
([Lewis et al., 2020](https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)).
The local rubric judge is G-Eval-style, not a reproduction of GPT-4 G-Eval
([Liu et al., 2023](https://aclanthology.org/2023.emnlp-main.153/)). Its score is not
evidence until agreement is measured on the same replies.
