# Evidence report: local @AmazonHelp support pipeline

## Executive result

The rebuilt system fixes the largest safety failure but does not yet establish a strong
intent model. On the audited 200-case gold set, escalation recall is 0.909 versus 0.309
for the keyword baseline. Intent macro-F1 is 0.781 versus 0.736, below the proposed
0.90 target. These are the values written by
`python main.py benchmark`; no other number is treated as
current.

The implementation is therefore a verified local prototype, not a production-ready
agent. Its chief remaining dependency is independent review of the prepared 400 training
examples through the blind review application. There are zero such consensus labels in
the current result.

## Problem framing: what good means

For `@AmazonHelp`, a useful first-response agent must do three things together: identify
the customer's primary need, avoid auto-handling cases with security, financial, legal,
or delivery-dispute risk, and provide a next step consistent with how Amazon historically
responded. The operating target is therefore high escalation recall first, followed by
intent macro-F1 and safe deflection. A fluent reply is not useful when routing is unsafe.

This prototype does not issue refunds, cancel orders, inspect accounts, promise delivery
times, authenticate customers, or replace a support specialist. It does not claim a
production SLA, a conformal guarantee, or unrestricted autonomous handling. Operational
dashboards are observability only and are not treated as quality evidence.

## What is implemented

### Supervision

Historical AmazonHelp replies provide the primary weak labels. Reply actions such as a tracking
question, return link, device troubleshooting, or account-security instruction provide
intent labels. A move to DM, phone, or chat provides escalation-positive supervision.
Ambiguous reply labels may be filled only when a four-source multiclass EM label model
reaches confidence 0.75. Its reply, resolution-act, issue, and request labeling functions
may abstain; it cannot overwrite a high-precision reply label. This expands coverage by
2,033 examples while conflicts remain candidates for human review.

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
escalation head's regularization is selected using three-fold average precision on its
training fold; the saved run selects `C=0.25` without class reweighting. The
primary escalation threshold maximizes precision subject to an exact one-sided 95%
recall lower bound of at least 0.85; a second threshold identifies only high-precision
learned transfers.
Low intent confidence abstains into escalation. This is a coverage–accuracy tradeoff,
not a conformal guarantee.

The raw TWCS reply graph is also followed backwards to construct up to four causally
prior customer/agent turns. The current inbound message and subsequent Amazon reply are
excluded by construction. This context exists for 50.6% of processed pairs and is shown
to reviewers and live sessions. A dual-view context gate, semantic-only head, and
nearest-prototype blend are fitted as candidate branches, but only activate after a
holdout improvement. In the saved run all three remain at zero weight: the prototype
gain was 0.0041 macro-F1, below the fixed 0.005 floor, and there are no human-reviewed
context labels. Thus context is available without silently degrading the scored model.

Public PII, account takeover, legal threats, delivery disputes, severe damage, financial
disputes, and prompt manipulation run through deterministic gates and cannot be vetoed.
Only an explicit routine self-service request backed by an intent-compatible retrieved
act may override a mid-range score; uncertainty and high-confidence risk still escalate.

### Retrieve then rewrite

The retriever combines BM25 with MiniLM cosine similarity using reciprocal-rank fusion.
Production searches an intent-filtered pool of 12 candidates so rare, compatible acts
are less likely to be hidden by the dominant general-support class; only the top three
records are returned as visible evidence.
It stores PII-stripped historical replies and one of eight resolution acts:

`ASK_TRACKING_STATUS`, `ASK_CARRIER`, `DM_WITH_ORDER_ID`, `LINK_ORDERS`,
`LINK_RETURNS`, `TROUBLESHOOT_DEVICE`, `APOLOGIZE_WAIT`, `GENERAL_SUPPORT`.

An RRF-weighted, deterministic vote over intent-compatible acts controls the next step.
In `local` mode, the act and up to two act-matched,
sanitized neighbors condition a revision-pinned Qwen2.5-1.5B-Instruct prompt. In default `offline`
mode, the same retrieved act selects a deterministic act-specific fallback. Both paths
then pass through URL, PII, false-promise, and length guardrails. Local output is rejected
before dispatch if it misses the act, copies a neighbor, requests sensitive data publicly,
uses a non-exact approved URL, or exceeds 280 characters; the trace exposes the fallback.
In the saved eight-act qualification, 3/8 local drafts were accepted and 5/8 used the
safe fallback. All final outputs met the deterministic checks; this validates containment,
not human-perceived reply quality.

### Reply evaluation

The old phrase counter has been removed. The optional local judge asks Qwen for JSON
scores on groundedness, empathy, actionability, and safety with the retrieved evidence
visible. It was validated on a stratified sample of 40 historical replies for which the
gold artifact contains human ratings; content hashes prove that the human and judge
scores refer to identical text. Overall Spearman correlation is -0.1609, quadratic
weighted kappa is 0.0000, and MAE is 0.6062. The model's 4.263 mean on 40 current system
drafts is consequently not a trustworthy quality score and is not a headline result.
The underlying human ratings are also limited: they came from one legacy annotator,
the annotator identity and repeat-rating protocol are unrecorded, and safety has zero
variance. The result proves the judge was checked; it does not validate the judge.

## Saved benchmark result

| Metric | Majority / always escalate | TF-IDF / keyword | Proposed |
|---|---:|---:|---:|
| Intent accuracy | 0.225 | 0.745 | 0.795 |
| Intent macro-F1 | 0.061 | 0.736 | 0.781 |
| Escalation accuracy | 0.550 | 0.520 | 0.575 |
| Escalate recall | 1.000 | 0.309 | 0.909 |
| Escalate precision | 0.550 | 0.630 | 0.571 |
| Exact reason match | 0.000 | 0.000 | 0.470 |
| Replies ≤280 characters | 1.000 | 1.000 | 1.000 |
| No direct PII echo | 1.000 | 1.000 | 1.000 |

Proposed-model 95% paired bootstrap intervals (2,000 iterations):

- intent macro-F1: 0.719–0.839;
- escalation recall: 0.850–0.959;
- escalation precision: 0.500–0.646.

Per-class proposed intent results:

| Intent | Precision | Recall | F1 |
|---|---:|---:|---:|
| Account and security | 0.917 | 0.629 | 0.746 |
| Damaged or wrong item | 0.833 | 0.857 | 0.845 |
| Feedback and general | 0.609 | 0.560 | 0.583 |
| Order tracking and delay | 0.796 | 0.867 | 0.830 |
| Product and technical | 0.786 | 0.880 | 0.830 |
| Refund and return | 0.800 | 0.914 | 0.853 |

Calibration was fitted on 975 weakly labeled intent examples and 870 reply-labeled
escalation examples. The fitted temperatures are 1.257 for intent and 0.756 for
escalation. The recall and high-precision thresholds are 0.2042 and 0.6268. Gold labels
were not used to choose them.

On the escalation calibration split, observed recall is 0.8947 and its exact one-sided
95% lower bound is 0.8623 over 323 positives. On gold, recall is 0.9091 and its
analogous lower confidence bound is 0.8507. This remains a test-set diagnostic rather
than a distribution-free safety guarantee.

On the gold diagnostic, 10-bin quantile ECE is 0.0630 and MCE is 0.1558. These are
descriptive test-set diagnostics, not calibration-fit inputs.

Exact paired tests are also saved. Proposed intent corrects 20 TF-IDF errors while
regressing on 10, but the exact two-sided McNemar p-value is 0.098737. Proposed routing
corrects 67 keyword decisions while regressing on 56; the corresponding accuracy
p-value is 0.367297. Neither supports a blanket superiority claim on n=200.

The held-out retrieval diagnostic exposes severe act imbalance rather than hiding it.
Top-3 macro act recall is 0.304 with a held-out reply-side weak-intent filter; the
12-candidate pool reaches 0.701 macro recall. The latter is a candidate-coverage and
upper-bound diagnostic, not end-to-end act accuracy.

## What is misleading about my headline number?

The most attractive number is 90.9% escalation recall, but it does not mean the agent is
90.9% accurate or production-ready. It comes with 57.1% escalation precision and an
87.5% escalation rate, so safety was purchased with substantial over-escalation. Its
bootstrap 95% interval is 85.0%–95.9%, and the one-sided lower bound is 85.1%.

The 200-case gold set was repeatedly consulted during development. IDs were excluded
from training and retrieval, but design and threshold changes were informed by aggregate
results and failures. It is therefore a development benchmark and may be optimistic.
A fresh sealed test set is required for a final generalization claim.

Intent macro-F1 improves from 0.736 to 0.781 over TF-IDF, but the paired exact test gives
`p=0.098737`; this is promising, not conclusive. The 100% length and direct-PII-echo
figures cover narrow deterministic invariants, not factual correctness or user preference.
Retrieval Recall@3 is dominated by `GENERAL_SUPPORT`; macro Recall@3 is only 0.304. The
0.701 macro coverage at 12 candidates uses held-out reply-side weak intent as an oracle
filter and is an upper-bound diagnostic. Finally, the saved headline uses deterministic
act-conditioned rewriting. The local Qwen qualification accepted only 3/8 drafts and
safely replaced the other five, so it is not evidence of superior generation quality.
The same model's apparent reply-judge mean is 4.263, but its same-reply Spearman
correlation with human scores is -0.161 and kappa is 0.000. Quoting 4.263 without those
agreement results would repeat the circular-evaluation failure this rebuild removed.

### What the result does and does not establish

It proves that, in this frozen checkout and audited gold set:

- the scored path uses a fitted semantic-plus-lexical representation, not keyword logit boosts;
- a learned, cost-sensitive decision plus hard gates reaches the intended escalation
  recall region;
- retrieved resolution acts change the generated next step;
- exact reason codes are used;
- causal context ingestion matches the live session format and cannot see future turns;
- unsupported auxiliary branches are automatically left inactive;
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

## Top five failure modes

1. **Routine cases are over-escalated.** There are 75 false-positive escalations and
   only 15 correctly auto-handled negatives. `GOLD-001` asks about repeated delivery
   notifications but is routed as a missing-delivery dispute. The recall-constrained
   threshold is intentionally low and historical DM supervision is not the same as the
   desired policy. Human policy labels and a precision-maximizing threshold under the
   recall constraint should reduce this.
2. **Some real delivery disputes still leak through.** Ten escalation positives are
   auto-handled; eight are tracking disputes. `GOLD-023` says a parcel was marked
   delivered but never received, yet receives risk 0.152 and a routine reason. The weak
   channel-transfer labels underrepresent policy risk. Add reviewed dispute labels and
   evaluate recall by risk subtype rather than adding another phrase rule.
3. **The account class is semantically inconsistent.** Only 22/35 account cases are
   recalled. Six become tracking and three become refund. `GOLD-117`, labelled account,
   is primarily a late Prime delivery; `GOLD-121` combines membership cancellation with
   a card charge. The taxonomy mixes membership, billing, and security. A blinded label
   audit should either split membership from security or document primary-intent rules.
4. **Feedback/general is a residual bucket.** It recalls 14/25 cases and has F1 0.583.
   `GOLD-184` is product feedback about an Echo Show image and is predicted technical;
   `GOLD-195` is feedback describing a late delivery and is predicted tracking. These
   are defensible disagreements, indicating taxonomy ambiguity rather than only model
   error. Multi-label annotations or an explicit `OTHER` policy would reduce forced labels.
5. **Historical action extraction is imbalanced and local rewriting is brittle.** Of
   500 held-out replies, 452 have weak act `GENERAL_SUPPORT`. Top-3 macro act recall is
   0.304, and Qwen accepted only 3/8 qualification drafts. The current safe fallback
   contains failures but reduces personalization. Human act labels, class-balanced
   retrieval evaluation, and a trained act reranker are the next experiment.

## What I would do with one more week

- **Day 1:** Blindly audit the six-intent policy and adjudicate ambiguous Prime,
  membership, billing, and delivery examples before changing any model.
- **Days 2–3:** Have two reviewers independently label the 400 selected training cases;
  resolve disagreements and preserve reviewer-level files and agreement statistics.
- **Day 4:** Fine-tune MiniLM only on consensus labels, then refit calibrated intent,
  escalation, reason, and act heads. Keep an ablation only if macro-F1 improves without
  reducing the escalation-recall lower bound.
- **Day 5:** Train and evaluate an act reranker on held-out human act labels; tighten
  local generation prompts using observed failure categories, not benchmark phrases.
- **Day 6:** Rate the same 40–50 system replies and evaluate local-judge agreement,
  including judge failure examples and dimension-level statistics.
- **Day 7:** Freeze a newly sampled 150–200-case audit set, run once, perform a clean-clone
  reproduction under 15 minutes, and update only claims supported by that final run.

## Decision log

1. Chose `@AmazonHelp` because it has enough multi-turn volume and diverse operational intents.
2. Used six intents to keep the policy explainable and obtain reasonable per-class support.
3. Made escalation a separate binary head because issue type and handling risk are different labels.
4. Kept hard gates for public PII, takeover, legal threats, and prompt manipulation because a model must not veto them.
5. Derived weak supervision primarily from agent replies because historical actions are closer to support outcomes than customer keywords.
6. Allowed weak labeling functions to abstain and used probabilistic aggregation rather than converting every rule match into truth.
7. Combined frozen MiniLM semantics with word and character TF-IDF because semantic embeddings and typo-sensitive lexical evidence fail differently.
8. Used a true reply-labelled calibration holdout and scalar temperature scaling instead of manual logit boosts.
9. Selected escalation thresholds under a recall lower-bound constraint because false negatives cost more than extra human review.
10. Used BM25 and MiniLM reciprocal-rank fusion because lexical identifiers and semantic similarity are both useful in noisy tweets.
11. Retrieved a resolution act and rewrote it instead of replaying a historical reply, reducing copied PII and irrelevant wording.
12. Defaulted to deterministic rewriting and made local Qwen optional because only 3/8 qualified local drafts passed without fallback.
13. Rejected CRAG, energy-OOD, and conformal headline claims because those methods are not validated on the scored decision path.
14. Froze gold artifacts by hash and preserved row-level predictions because reproducible evidence is more valuable than a dashboard screenshot.

## Human-review evidence gate

Two independent reviewers should run `python main.py review --port 8001` under different
reviewer IDs. The application hides weak labels and predictions and writes separate,
source-hashed files under `data/reviews/`. `scripts/adjudicate_reviews.py` emits only
exact, non-ambiguous consensus into `data/training/human_consensus_labels.csv`; disputed
rows stay separate. Contrastive tuning cannot start until at least 48 consensus labels
cover all six intents. Separately, the `/ratings` desk collects blind ratings on the exact
50 drafts. The current 40-case historical-reply validation meets the same-text
requirement but shows that the 1.5B judge is unreliable. Independent ratings of current
system drafts remain the human-only evidence task before any reply-quality claim.

## Reproduction and verification

The saved headline artifact was produced without an API key:

```powershell
.\.venv\Scripts\python.exe main.py benchmark
```

The first run trains and writes a fingerprinted cache; a second run must reproduce the
same metrics while reporting `trained_pipeline_cache_hit: true`. Rebuild historical
context from the source dataset with
`.\.venv\Scripts\python.exe scripts\build_conversation_context.py`.

The gold CSV and JSONL hashes are frozen in `data/golden/MANIFEST.json` and verified
before every benchmark. Retrieval and local generation diagnostics run with:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --queries 500
.\.venv\Scripts\python.exe scripts\qualify_local_generator.py
.\.venv\Scripts\python.exe main.py judge-eval --judge-limit 40 --judge-batch-size 8
```

The default offline generation path, local API lifecycle, label audit, safety checks,
and row-level evidence are covered by the repository test suite. The slower local Qwen
rewrite and rubric judge are opt-in. Their content-addressed evidence is retained, while
the failed judge validation prevents its numerical score from becoming a quality claim.

## References and claim boundaries

- MiniLM is used as a local sentence embedding model as documented in its
  [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- Temperature scaling is a single fitted parameter following
  [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html).
- The cross-fitted weak-label audit is a conservative self-confidence filter inspired
  by confident learning; it is not claimed as a full Cleanlab implementation
  ([Northcutt et al., 2021](https://arxiv.org/abs/1911.00068)).
- A local pairwise contrastive training path is implemented as an explicit ablation,
  but disabled by default because its weak-label run reduced escalation recall. SetFit
  is not claimed as implemented; it remains a candidate after real labels exist
  ([Tunstall et al., 2022](https://arxiv.org/abs/2209.11055)).
- Retrieval grounding is inspired by, but not claimed as an exact implementation of,
  [Lewis et al. (2020)](https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html).
- The local rubric is G-Eval-style and not a reproduction of GPT-4 G-Eval
  ([Liu et al., 2023](https://aclanthology.org/2023.emnlp-main.153/)).
