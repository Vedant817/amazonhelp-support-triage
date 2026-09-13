# @AmazonHelp support-agent assignment

This repository implements a local, evidence-first support pipeline over historical
`@AmazonHelp` conversations. It does not call a hosted model API.

The current system is:

1. non-vetoable policy gates for public PII, takeover, legal threats, delivery disputes,
   severe damage, financial disputes, and prompt manipulation;
2. reply-side labels plus conservative probabilistic coverage expansion from four
   abstaining weak sources;
3. MiniLM semantic features fused with word and character TF-IDF features;
4. out-of-fold confidence filtering of the noisiest 5% of intent weak labels;
5. separate intent and escalation heads with holdout temperature scaling, with
   escalation regularization chosen by training-only three-fold cross-validation;
6. confidence abstention plus an exact-confidence-bound escalation threshold;
7. causally reconstructed prior turns plus a validation-gated dual-view classifier;
8. BM25 + MiniLM retrieval fused with reciprocal-rank fusion over an intent-filtered
   12-neighbor candidate pool, with only the top three records exposed as evidence;
9. a resolution act extracted from historical replies;
10. either local Qwen rewriting or deterministic act-conditioned offline rewriting;
11. URL, PII, false-promise, and 280-character guardrails.

## Verified status

`results/headline_results.json` is the only headline source. The saved artifact was
produced with `python main.py benchmark`; the optional CPU-heavy local baseline was not
run in this artifact. On the audited 200-case gold set:

| System | Intent macro-F1 | Escalate recall | Escalate precision | Reason exact match |
|---|---:|---:|---:|---:|
| Majority + always escalate | 0.061 | 1.000 | 0.550 | 0.000 |
| TF-IDF + keyword escalation + retrieve/replay | 0.736 | 0.309 | 0.630 | 0.000 |
| Probabilistic hybrid dual-head + gates + retrieve/rewrite | **0.781** | **0.909** | 0.571 | **0.470** |

The proposed system's bootstrap 95% intervals are:

- intent macro-F1: 0.719–0.839;
- escalation recall: 0.850–0.959;
- escalation precision: 0.500–0.646.

The exact one-sided 95% lower confidence bound for gold escalation recall is 0.8507.
That is a post-selection test-set diagnostic, not a distribution-free guarantee. The
independent reply-labeled calibration split meets its selection constraint (lower bound
0.8623 over 323 positives), but that is historical channel-transfer supervision rather
than final policy evidence.

These results are materially safer than the keyword baseline on escalation recall, but
they do **not** meet the intended intent-quality or escalation-precision targets. There
are currently zero independently human-reviewed training labels. The prepared 400-row
queue is `data/training/labels_for_human_review.csv`; it is reviewed through the blind
local review desk, not edited in place. Blank or single-reviewer rows are never treated
as consensus labels.

No LLM-judge score or judge–human agreement is claimed in the default run. Those fields
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
Face cache and are not committed. The fitted pipeline is cached under
`models/pipeline_cache/` using a fingerprint of the data, configuration, and relevant
source files; changing any of them forces retraining.

Rebuild the causal conversation artifact from the original TWCS data with:

```powershell
.\.venv\Scripts\python.exe scripts\build_conversation_context.py
```

Regenerate the active-learning review batch only while its review fields are empty:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_review_labels.py --force
```

Start the blind label and exact-reply rating interfaces with:

```powershell
.\.venv\Scripts\python.exe main.py review --port 8001
```

Each reviewer uses a different ID. The UI hides weak labels, model predictions, gold
labels, and local-judge scores. After two reviewers complete the same cases, create the
training consensus and reply-rating files with:

```powershell
.\.venv\Scripts\python.exe scripts\adjudicate_reviews.py `
  data/reviews/training-review-reviewer-a.csv `
  data/reviews/training-review-reviewer-b.csv

.\.venv\Scripts\python.exe scripts\aggregate_reply_ratings.py `
  data/reviews/reply-rating-reviewer-a.csv `
  data/reviews/reply-rating-reviewer-b.csv
```

The contrastive ablation is hard-gated: `--contrastive-pairs 600` fails unless at least
48 independent consensus labels cover all six intents. It fine-tunes only on those human
labels, never on weak labels. Keep it only if the frozen benchmark improves without a
routing-safety regression.

The API defaults to deterministic offline generation so it can be reproduced without
loading a generative model. To use the pinned local Qwen model:

```powershell
.\.venv\Scripts\python.exe main.py serve --generation-mode local
```

The dashboard reads its runtime model IDs, revisions, device, corpus size, generation
backend, and calibration values from `/api/v1/health`. Each triage response includes
the complete intent distribution and a measured six-stage execution trace covering
classification, risk scoring, retrieval, routing policy fusion, generation, and
pre-dispatch guardrails.

Qualify the keyless local rewrite separately before a demo:

```powershell
.\.venv\Scripts\python.exe scripts\qualify_local_generator.py
```

The pinned generator is Qwen2.5-1.5B-Instruct. A local draft is accepted only when it
performs the retrieved act, is not a near-verbatim historical reply, requests no public
sensitive details, passes the URL/commitment/PII guard, and fits the character budget.
The API trace reports the actual backend and any deterministic fallback reason. In the
saved eight-act CPU qualification, 3 local drafts were accepted and 5 safely fell back;
all eight final outputs passed the act, guardrail, and 280-character checks. Treat local
rewriting as an optional demo path, not as proven reply-quality improvement.

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
- Historical context follows `in_response_to_tweet_id` backwards and contains only
  already-observed turns. It covers 50.6% of the processed conversations.
- High-precision intent weak labels are inferred from AmazonHelp's reply. A multiclass
  EM label model combines reply, resolution-act, and customer-side labeling functions
  only to fill reply-label abstentions at confidence >=0.75; it never overwrites reply labels.
- Whether Amazon moved a conversation to DM/phone/chat supplies the escalation weak label.
- Ambiguous historical replies remain unlabeled.
- `scripts/audit_gold_labels.py` applies only four documented policy corrections; see
  `data/golden/LABEL_AUDIT.md`.
- Independent reviewers can override weak labels and add reason/act labels in separate,
  source-hashed reviewer files generated by the blind review desk.
- `data/golden/MANIFEST.json` freezes both gold representations by SHA-256; the benchmark
  refuses to run if either changes.

## Evaluation contract

Headline metrics are macro-F1, escalation recall/precision, exact reason-code match,
bootstrap intervals, and reply safety invariants. The benchmark also writes:

- `results/golden_predictions.csv` — row-level evidence;
- `results/failure_examples.csv` — the first ten actual gold errors;
- `results/human_rating_template.csv` — same system replies for independent rating;
- `results/baseline_comparison.csv` — concise comparison table.
- `results/tough_scenario_results.csv` — row-level difficult-case routing evidence.
- `results/retrieval_evaluation.json` — held-out, per-act retrieval diagnostics.
- `results/local_generator_qualification.json` — real local generation acceptance and fallback evidence.

ROUGE/BLEU are not treated as reply-quality metrics. Conformal risk guarantees, energy
OOD, CRAG, and a keyword-based “LLM judge” are not part of the scored production path.

## Method scope

MiniLM maps text into 384-dimensional vectors suitable for semantic similarity
([model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)). This
implementation fuses those frozen embeddings with learned word and character TF-IDF
features and filters low-confidence weak labels using out-of-fold predictions. A local
contrastive fine-tuning ablation is available through `--contrastive-pairs`, but is off
by default: on weak labels it produced only a small intent gain and reduced escalation
recall. It is not claimed as SetFit. SetFit remains appropriate after independent labels exist
([Tunstall et al., 2022](https://arxiv.org/abs/2209.11055)). Temperature scaling follows
the single-parameter calibration method of
[Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html).
The weak-label filter is inspired by confident learning but is not claimed as a full
implementation ([Northcutt et al., 2021](https://arxiv.org/abs/1911.00068)).

The final escalation head uses `C=0.25`, selected by three-fold cross-validated
average precision (0.6118) on the training fold only. Context, a semantic-only head,
and a nearest-prototype ensemble are implemented as holdout-gated candidates. They are
all assigned zero weight in the saved run: the prototype gain was only 0.0041 macro-F1,
below the pre-set 0.005 materiality floor, and no human-reviewed context labels exist.
This prevents complexity that did not validate from changing the scored path.

The paired exact tests do not establish a statistically significant accuracy difference
on this 200-case set: proposed-vs-TF-IDF intent has `p=0.098737`, and proposed-vs-keyword
escalation accuracy has `p=0.367297`. The safety case therefore leads with escalation
recall and its interval, not an unsupported general-superiority claim.

Held-out act retrieval is strongly imbalanced. Unfiltered top-3 macro recall is 0.211.
With the intent-filtered 12-neighbor diagnostic pool it is 0.701, while top-3 macro
recall is 0.304. The filter uses held-out reply-side weak intent and is explicitly an
upper-bound diagnostic; it is not a production accuracy claim.

The retrieve-then-rewrite design is inspired by retrieval-augmented generation, but is
described by what it actually does rather than claimed as the exact RAG training method
([Lewis et al., 2020](https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)).
The local rubric judge is G-Eval-style, not a reproduction of GPT-4 G-Eval
([Liu et al., 2023](https://aclanthology.org/2023.emnlp-main.153/)). Its score is not
evidence until agreement is measured on the same replies.
