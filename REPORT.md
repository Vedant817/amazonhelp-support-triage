# Evidence report: local @AmazonHelp support pipeline

## Executive result

The rebuilt system fixes the largest safety failure but does not yet establish a strong
intent model. On the audited 200-case gold set, escalation recall is 0.891 versus 0.309
for the keyword baseline. Intent macro-F1 is 0.748 versus 0.673, below the proposed
0.90 target. These are the values written by
`python main.py benchmark`; no other number is treated as
current.

The implementation is therefore a verified local prototype, not a production-ready
agent. Its chief remaining dependency is independent review of the prepared 400 training
examples. There are zero such reviewed labels in the current result.

## What is implemented

### Supervision

Historical AmazonHelp replies provide weak labels. Reply actions such as a tracking
question, return link, device troubleshooting, or account-security instruction provide
intent labels. A move to DM, phone, or chat provides escalation-positive supervision.
Ambiguous replies stay unlabeled. No customer-keyword rules create proposed-model labels.

Gold tweet IDs are removed before any labeling, training, calibration, or retrieval. Four
clear policy contradictions were corrected before this model was evaluated; the exact
IDs and rationales are in `data/golden/LABEL_AUDIT.md`.

### Learned decisions

The local `sentence-transformers/all-MiniLM-L6-v2` encoder produces normalized
384-dimensional embeddings. Word and character TF-IDF features preserve lexical and
typo-level evidence that sentence embeddings can smooth away. A five-fold out-of-fold
self-confidence audit removes the lowest-confidence 5% within each intent class before
the final head is fitted. Separate regularized heads then learn:

- the six-way intent;
- the binary escalation decision.

Each head has a separately fitted scalar temperature on a reply-labeled holdout. The
primary escalation threshold is the highest holdout threshold meeting 0.85 positive
recall; a second threshold identifies only high-precision learned transfers.
Low intent confidence abstains into escalation. This is a coverage–accuracy tradeoff,
not a conformal guarantee.

Public PII, account takeover, legal threats, delivery disputes, severe damage, financial
disputes, and prompt manipulation run through deterministic gates and cannot be vetoed.
Only an explicit routine self-service request backed by an intent-compatible retrieved
act may override a mid-range score; uncertainty and high-confidence risk still escalate.

### Retrieve then rewrite

The retriever combines BM25 with MiniLM cosine similarity using reciprocal-rank fusion.
It stores PII-stripped historical replies and one of eight resolution acts:

`ASK_TRACKING_STATUS`, `ASK_CARRIER`, `DM_WITH_ORDER_ID`, `LINK_ORDERS`,
`LINK_RETURNS`, `TROUBLESHOOT_DEVICE`, `APOLOGIZE_WAIT`, `GENERAL_SUPPORT`.

An RRF-weighted, deterministic vote over intent-compatible acts controls the next step.
In `local` mode, the act and two
sanitized neighbors condition a pinned Qwen2.5-0.5B-Instruct prompt. In default `offline`
mode, the same retrieved act selects a deterministic act-specific fallback. Both paths
then pass through URL, PII, false-promise, and length guardrails.

### Reply evaluation

The old phrase counter has been removed. The optional local judge asks Qwen for JSON
scores on groundedness, empathy, actionability, and safety with the retrieved evidence
visible. Default results contain no judge score. Human agreement is computed only when
human ratings join to the exact judged reply IDs; the statistic is Spearman correlation
plus quadratic weighted kappa.

## Saved benchmark result

| Metric | Majority / always escalate | TF-IDF / keyword | Proposed |
|---|---:|---:|---:|
| Intent accuracy | 0.225 | 0.685 | 0.765 |
| Intent macro-F1 | 0.061 | 0.673 | 0.748 |
| Escalation accuracy | 0.550 | 0.520 | 0.570 |
| Escalate recall | 1.000 | 0.309 | 0.891 |
| Escalate precision | 0.550 | 0.630 | 0.570 |
| Exact reason match | 0.000 | 0.000 | 0.470 |
| Replies ≤280 characters | 1.000 | 1.000 | 1.000 |
| No direct PII echo | 1.000 | 1.000 | 1.000 |

Proposed-model 95% paired bootstrap intervals (2,000 iterations):

- intent macro-F1: 0.683–0.806;
- escalation recall: 0.829–0.945;
- escalation precision: 0.500–0.643.

Per-class proposed intent results:

| Intent | Precision | Recall | F1 |
|---|---:|---:|---:|
| Account and security | 0.909 | 0.571 | 0.702 |
| Damaged or wrong item | 0.800 | 0.800 | 0.800 |
| Feedback and general | 0.542 | 0.520 | 0.531 |
| Order tracking and delay | 0.800 | 0.889 | 0.842 |
| Product and technical | 0.719 | 0.920 | 0.807 |
| Refund and return | 0.784 | 0.829 | 0.806 |

Calibration was fitted on 763 reply-labeled intent examples and 870 reply-labeled
escalation examples. The fitted temperatures are 1.304 for intent and 0.867 for
escalation. The recall and high-precision thresholds are 0.2337 and 0.6430. Gold labels
were not used to choose them.

On the gold diagnostic, 10-bin quantile ECE is 0.0498 and MCE is 0.0993. These are
descriptive test-set diagnostics, not calibration-fit inputs.

## What the result proves—and does not

It proves that, in this frozen checkout and audited gold set:

- the scored path uses a fitted semantic-plus-lexical representation, not keyword logit boosts;
- a learned, cost-sensitive decision plus hard gates reaches the intended escalation
  recall region;
- retrieved resolution acts change the generated next step;
- exact reason codes are used;
- output length and direct PII-echo checks pass all 200 cases;
- uncertainty, prediction rows, and failures are saved for inspection.

It does not prove:

- 0.90 intent macro-F1;
- acceptable escalation precision or deflection;
- 0.70 reason exact match;
- human preference or judge reliability;
- production latency (this environment has a CPU-only PyTorch build);
- a conformal, CRAG, or energy-OOD guarantee;
- that the 400 prepared examples were human reviewed.

## Next evidence gate

Two independent reviewers should fill `reviewed_intent` and `reviewed_escalate` in
`data/training/labels_for_human_review.csv`, resolve disagreements without consulting
gold predictions, and rerun the benchmark. Only after those labels exist is SetFit-style
encoder fine-tuning justified. Separately, reviewers should score the exact 50 replies in
`results/human_rating_template.csv`; only those matching reply IDs may be compared with
the local judge.

## Reproduction and verification

The saved headline artifact was produced without an API key:

```powershell
.\.venv\Scripts\python.exe main.py benchmark
```

The default offline generation path, local API lifecycle, label audit, safety checks,
and row-level evidence are covered by the repository test suite. The slower local Qwen
rewrite and rubric judge are opt-in; no judge result is included in the saved headline
artifact.

## References and claim boundaries

- MiniLM is used as a local sentence embedding model as documented in its
  [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- Temperature scaling is a single fitted parameter following
  [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html).
- The cross-fitted weak-label audit is a conservative self-confidence filter inspired
  by confident learning; it is not claimed as a full Cleanlab implementation
  ([Northcutt et al., 2021](https://arxiv.org/abs/1911.00068)).
- SetFit is not claimed as implemented; it is the planned fine-tuning method after real
  labels exist ([Tunstall et al., 2022](https://arxiv.org/abs/2209.11055)).
- Retrieval grounding is inspired by, but not claimed as an exact implementation of,
  [Lewis et al. (2020)](https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html).
- The local rubric is G-Eval-style and not a reproduction of GPT-4 G-Eval
  ([Liu et al., 2023](https://aclanthology.org/2023.emnlp-main.153/)).
