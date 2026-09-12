# Research-backed plan to improve triage performance

> Implementation update (2026-09-12): uncertainty/diversity review selection,
> conservative probabilistic weak-label coverage expansion, exact recall-bound threshold
> selection, and a local contrastive ablation are implemented. The active batch contains
> 400 leak-free rows, but still has zero completed human reviews. The default contrastive
> pair count is zero because the weak-label ablation reduced escalation recall.

## Executive conclusion

The next improvement should not be another keyword layer or a larger zero-shot
model. The dominant limitation is supervision: the current intent head learns
from reply-derived weak labels, no human-reviewed training labels have been
incorporated, and historical requests for a DM are only an imperfect proxy for
the assignment's `must_escalate` policy.

The most defensible next system is:

1. select high-value examples with uncertainty and diversity;
2. obtain human labels for intent, escalation reason, and resolution act;
3. combine the remaining weak signals with a probabilistic label model;
4. fine-tune the local MiniLM encoder with supervised contrastive learning and
   separate intent, escalation, reason, and act heads; and
5. choose the escalation threshold on an independent calibration set so that a
   one-sided confidence bound, not only a point estimate, meets the recall goal.

This remains fully local. It does not require a hosted model or an API key.

## What the present evidence says

The current saved benchmark reports intent macro-F1 `0.7812`, escalation recall
`0.9000`, escalation precision `0.5723`, reason exact match `0.47`, and ECE
`0.0630`. The escalation recall point estimate is encouraging, but its exact
one-sided 95% lower bound is `0.8399`. It therefore does **not** establish a
recall guarantee of 0.85.

There are 31,942 leak-free historical customer/reply pairs, but the weak intent
labels are highly imbalanced: 5,778 tracking examples versus only 191 account
and security examples. Another 23,552 pairs have no intent label. That makes
better use of unlabeled data and scarce human annotation more valuable than
adding capacity to the present classifier.

The existing 200-row gold set has also participated in repeated development
decisions. Adaptive reuse of a holdout can overfit the holdout even when the
training code never directly reads its labels.[^1] It should now be treated as a
development benchmark. Final claims require a newly labeled, sealed audit set.

## Recommended method

### 1. Replace random review sampling with active, diversity-aware sampling

The current review sheet is a stratified random sample of weakly labeled rows.
It should instead prioritize examples that can change the decision boundary.
BADGE is a relevant batch active-learning method because it selects examples
that are both uncertain and diverse, rather than selecting many near-duplicate
uncertain examples.[^2]

For every candidate `x`, compute:

```text
uncertainty(x)     = H(p_intent(x)) / log(6)
label_suspicion(x) = 1 - p_intent(weak_label | x)
route_margin(x)    = 1 - 2 * abs(p_escalate(x) - threshold)
disagreement(x)    = variance of semantic, lexical, retrieval, and rule votes
```

Select within class and routing strata using gradient embeddings if the joint
encoder is already trainable; otherwise use MiniLM embeddings with k-means++ or
farthest-first selection. Deliberately oversample:

- account/security and feedback/general;
- examples near the escalation threshold;
- semantic-versus-lexical disagreements;
- weak-label conflicts and suspected label errors; and
- underrepresented escalation reason codes and resolution acts.

The annotation schema should contain `intent`, `must_escalate`, `reason_code`,
`resolution_act`, `ambiguous`, and reviewer notes. Two reviewers should label a
small overlap subset, with disagreements adjudicated and agreement reported.

If only 400 new labels are affordable, use approximately 280 for training, 60
for model selection/calibration, and 60 as a sealed smoke-test set. This is
enough to compare iterations, but too small for a strong final safety claim. A
better target is 600-800 labels, preserving at least 200-300 examples for an
untouched, policy-stratified final audit.

### 2. Model weak supervision instead of treating rules as ground truth

Snorkel's label model was designed to combine multiple noisy labeling functions
while estimating their accuracies and dependencies.[^3] This is a closer fit to
the repository than a single ordered keyword cascade.

Create independent labeling functions that may abstain:

- historical reply act (`DM_WITH_ORDER_ID`, `LINK_RETURNS`, and so on);
- destination of a policy URL;
- channel transfer or request for private information;
- customer-side security, legal, damage, and delivery-dispute signals;
- nearest-neighbor intent and act votes;
- semantic prototype similarity; and
- conversation metadata, where available and leak-safe.

Fit separate probabilistic label models for intent, historical channel transfer,
and resolution act. Do **not** define `must_escalate` as “Amazon asked for a DM.”
That is an observed historical behavior, while escalation is the target policy.
Use human policy labels to learn the latter and retain channel transfer as a
feature or auxiliary target.

Confident Learning can rank likely label errors by estimating the joint
distribution of noisy and latent labels.[^4] Use out-of-fold probabilities to
produce a review queue; do not automatically delete every flagged row. The
existing lowest-loss pruning can remain as an ablation, but it should not be
presented as equivalent to Confident Learning.

### 3. Fine-tune the local encoder after the labels improve

SetFit contrastively fine-tunes a sentence-transformer body before fitting its
classification head and was designed for label-efficient text
classification.[^5] It is a practical, API-free upgrade from frozen MiniLM
embeddings.

Fine-tune `all-MiniLM-L6-v2` locally with same-label positives and hard negatives
drawn from confused class pairs. Then train masked task heads:

```text
L = L_intent + lambda_e * L_escalate
               + lambda_r * L_reason + lambda_a * L_act
```

Each loss is evaluated only where that label is known. Use class-balanced batch
sampling, early stopping on macro-F1 and escalation recall, and out-of-fold
predictions for downstream calibration. Keep the lexical heads as an ensemble
member until an ablation proves they no longer help.

The contrastive intent-detection literature specifically supports pulling
same-intent utterances together and separating different intents in few-shot
settings.[^6] This is especially relevant to the current account/security and
feedback/general confusions.

There is also a large pool of unlabeled in-domain text. Task-adaptive or
domain-adaptive language-model pretraining can improve downstream classification
before supervised fine-tuning.[^7] Run this only after the clean-label experiment:
it costs more compute and cannot repair an incorrect escalation policy.

### 4. Turn escalation into a constrained decision problem

The operating objective is not maximum binary accuracy. Let `s(x)` be the
calibrated escalation score and `t` the threshold. Choose:

```text
maximize_t    auto_handle_rate(t)
subject to    lower_95_bound(recall_escalate(t)) >= 0.85
```

Neyman-Pearson classification provides an umbrella method for selecting a
threshold with high-probability control of a chosen class-conditional error,
using a held-out calibration sample and an ordered list of scores.[^8] Here the
controlled error is the false-negative rate for tickets that require escalation.
Hard PII, account-takeover, legal, and payment-data gates remain non-overridable.

The calibration set must be independent of model fitting and threshold search.
Using a one-sided exact Clopper-Pearson lower bound at alpha `0.05`:

- the present 98/110 recalled positives have a lower bound of about `0.829`;
- an observed 90% recall needs roughly 121/134 positives before its lower bound
  reaches 0.85;
- an observed 92% needs roughly 63/68; and
- an observed 95% needs roughly 38/40.

Those are illustrative minimums at the stated observed proportions, not a power
analysis or a guaranteed future sample size. Plan the calibration set using the
lowest plausible observed recall, and reserve a separate final test set.

Temperature scaling can still calibrate the score on an in-distribution holdout,
but calibration can deteriorate under dataset shift.[^9] Report reliability by
time slice and issue type; abstain or escalate when the example is far from the
calibration distribution.

### 5. Predict escalation reasons and retrieval acts directly

Reason exact match is currently limited because intent and route are mapped into
reason strings. Train a closed-set reason head, conditional on escalation, from
human labels. Allow `OTHER_REVIEW_REQUIRED` as an explicit fallback rather than
inventing a reason.

For generation, keep hybrid retrieval but evaluate it as a retrieval system.
Label relevant historical resolutions and report Recall@3 and MRR. A small local
cross-encoder can rerank top candidates using positive same-act pairs and hard
negative different-act pairs. The generator should receive the selected act and
PII-stripped examples; it must not treat a retrieved historical promise as
customer-specific fact.

### 6. Use robust-loss methods only when the audit justifies them

Generalized cross entropy[^10] and co-teaching[^11] are established approaches for
training deep networks with noisy labels. Their headline experiments focus
largely on synthetic image-label noise, so they are secondary candidates here.
First estimate real error patterns in the actively reviewed sample. Test one of
these losses only if substantial residual noise remains after probabilistic weak
supervision.

Positive-unlabeled learning is tempting for the 27,596 rows without escalation
labels. Non-negative PU learning provides a principled objective for positive
and unlabeled data.[^12] However, historical DM requests are likely a biased
subset of all policy escalations, violating the assumption that labeled
positives represent positives generally. Treat PU learning as a later ablation,
not the main route.

## Experiment sequence and stopping rules

Run paired experiments on fixed folds so changes are attributable:

| ID | Change from previous row | Decision criterion |
|---|---|---|
| M0 | Current hybrid model | Reproduce saved metrics exactly |
| M1 | Active human labels | Macro-F1 and reason exact improve; no recall regression |
| M2 | Probabilistic weak-label model | Beats hard weak labels on reviewed examples |
| M3 | SetFit contrastive encoder | Paired bootstrap improvement in macro-F1 |
| M4 | Joint reason and act heads | Better reason exact and retrieval Recall@3 |
| M5 | NP-calibrated threshold | Recall lower bound meets target on calibration data |
| M6 | Optional task-adaptive pretraining | Keep only if gain exceeds variance and latency budget |

Use nested cross-validation or a fixed development split for M1-M6. Compare
intent predictions with paired bootstrap confidence intervals and McNemar's test;
report per-class recall and confusion matrices. Report escalation recall with an
exact one-sided confidence bound, plus precision and auto-handle rate. Open the
sealed test set once, after the method and threshold are frozen.

## Realistic expected outcome

The following are experiment targets, **not predicted or guaranteed results**:

| Metric | Current | Next experiment target |
|---|---:|---:|
| Intent macro-F1 | 0.748 | 0.80-0.86 on a new sealed set |
| Escalation recall | 0.891 point estimate | lower 95% bound >= 0.85 |
| Escalation precision | 0.570 | 0.65-0.75 while satisfying recall constraint |
| Reason exact match | 0.470 | 0.60-0.75 |
| PII/length guardrails | 1.000 | retain 1.000 |

The largest likely gain comes from correcting the target definition and adding
high-information human labels. SetFit and adaptive pretraining should be judged
only after that foundation exists. If M1 does not materially outperform M0, the
failure is evidence that the label taxonomy or annotation policy needs revision,
not a signal to add more rules.

## Methods to avoid as headline claims

- more keyword logit boosts;
- a larger zero-shot local LLM without measured probability quality;
- CRAG without a trained retrieval evaluator;
- conformal or risk-control language when the method is not on the scored path;
- accuracy as the primary escalation metric;
- a judge score not validated against humans on the same system replies; and
- further adaptive tuning against the existing 200-row gold set.

## Sources

[^1]: Dwork et al., “The Reusable Holdout: Preserving Validity in Adaptive Data Analysis,” NeurIPS 2015. https://papers.neurips.cc/paper_files/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html
[^2]: Ash et al., “Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds,” ICLR 2020. https://www.microsoft.com/en-us/research/publication/deep-batch-active-learning-by-diverse-uncertain-gradient-lower-bounds/
[^3]: Ratner et al., “Snorkel: Rapid Training Data Creation with Weak Supervision,” PVLDB 2017. https://www.vldb.org/pvldb/vol11/p269-ratner.pdf
[^4]: Northcutt, Jiang, and Chuang, “Confident Learning: Estimating Uncertainty in Dataset Labels,” JAIR 2021. https://arxiv.org/abs/1911.00068
[^5]: Tunstall et al., “Efficient Few-Shot Learning Without Prompts,” 2022. https://arxiv.org/abs/2209.11055
[^6]: Zhang et al., “Few-Shot Intent Detection via Contrastive Pre-Training and Fine-Tuning,” EMNLP 2021. https://aclanthology.org/2021.emnlp-main.144/
[^7]: Gururangan et al., “Don't Stop Pretraining: Adapt Language Models to Domains and Tasks,” ACL 2020. https://aclanthology.org/2020.acl-main.740/
[^8]: Tong, Feng, and Li, “Neyman-Pearson Classification Algorithms and NP Receiver Operating Characteristics,” Science Advances 2018; and Tong et al., “Neyman-Pearson Classification: Parametrics and Sample Size Requirement,” JMLR 2020. https://pmc.ncbi.nlm.nih.gov/articles/PMC5804623/ and https://jmlr.org/papers/v21/18-577.html
[^9]: Ovadia et al., “Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift,” NeurIPS 2019. https://proceedings.neurips.cc/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf
[^10]: Zhang and Sabuncu, “Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels,” NeurIPS 2018. https://proceedings.neurips.cc/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf
[^11]: Han et al., “Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels,” NeurIPS 2018. https://papers.nips.cc/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html
[^12]: Kiryo et al., “Positive-Unlabeled Learning with Non-Negative Risk Estimator,” NeurIPS 2017. https://papers.neurips.cc/paper/2017/file/7cce53cf90577442771720a370c3c723-Paper.pdf
