# @AmazonHelp support-agent prototype

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

## Install and run

Prerequisite: Python 3.11 or 3.12. The checked-in dependency versions are pinned in
`requirements.txt`. Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

The first learned run downloads the pinned
`sentence-transformers/all-MiniLM-L6-v2` revision. Model files use the normal Hugging
Face cache and are not committed. The fitted pipeline is cached under
`models/pipeline_cache/` using a fingerprint of the data, configuration, and relevant
source files; changing any of them forces retraining.

> **First-run note:** the initial fit performs MiniLM encoding, TF-IDF fitting, and
> cross-validated regularization/model-selection searches. On a CPU-only machine this
> can take 10–30 minutes. Every later start reuses the cache in seconds. Model-backed
> endpoints return `503` until the fit completes.

The API defaults to deterministic offline generation so it can be reproduced without
loading a generative model. To use the pinned local Qwen model:

```powershell
.\.venv\Scripts\python.exe main.py serve --generation-mode local
```

## Decision console (UI)

Open <http://127.0.0.1:8000/> after starting the server. The console shows:

- a live runtime strip (encoder, reply backend, retrieval, calibration, corpus, device);
- a composed customer-message form with one-click demo cases and a `Ctrl` + `Enter` shortcut;
- the decision banner (escalate vs auto-reply) with the exact routing reason code;
- four headline metrics (confidence, risk, threshold, latency);
- the guarded draft reply with a 280-character budget and copy-to-clipboard;
- a measured six-stage execution trace and the full intent distribution;
- retrieved historical evidence with RRF / dense / BM25 scores;
- the pre-dispatch policy result and the complete raw API response.

The dashboard reads its runtime model IDs, revisions, device, corpus size, generation
backend, and calibration values from `/api/v1/health`. Each triage response includes
the complete intent distribution and a measured six-stage execution trace covering
classification, risk scoring, retrieval, routing policy fusion, generation, and
pre-dispatch guardrails.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

The suite covers the safety gates, policy guardrails, retrieval, generation,
dual-head calibration, and multi-turn session handling. A hostile end-to-end
smoke script is included for a running server:

```powershell
.\.venv\Scripts\python.exe scripts_smoke.py http://127.0.0.1:8000
```

## API

- `GET /api/v1/health` reports actual model temperatures, threshold, corpus size, and generation mode.
- `POST /api/v1/confidence` returns calibrated intent probabilities and the abstention decision.
- `POST /api/v1/triage` returns intent, learned escalation risk, exact reason code, resolution act, PII-stripped retrieval evidence, guarded draft, and latency.
- `GET /metrics` exposes operational counters. These are observability, not quality evidence.
- All model-backed endpoints return `503` while the pipeline is still warming up.

## Scores

Reproduce with `python scripts_score.py --holdout {random,temporal,review}`. Each
protocol writes its own leak-guard file and verifies, twice, that the evaluation
rows were actually excluded from training; a mismatch aborts the run rather than
reporting a contaminated score.

### Results by protocol

| Metric | `random` (expected traffic) | `temporal` (drift) | `review` (hardest rows) |
|---|---:|---:|---:|
| Evaluation rows | 1,268 | 1,268 | 382 |
| Intent accuracy | **0.8170** | 0.8446 | 0.5241 |
| 95% CI | [0.7957, 0.8383] | [0.8249, 0.8636] | [0.4655, 0.5828] |
| Intent macro-F1 | **0.6142** | 0.6217 | 0.2731 |
| Intent macro precision / recall | 0.7049 / 0.5637 | 0.6942 / 0.5808 | 0.2998 / 0.2852 |
| Deployment accuracy (traffic prior) | **0.8174** | 0.8300 | 0.6152 |
| ECE / MCE | **0.0277 / 0.0956** | 0.0424 / 0.0924 | 0.1848 / 0.5085 |
| Escalation recall (one-sided 95% lower bound) | 0.9268 (0.8216) | 0.7679 (0.6565) | 0.9375 (0.8463) |
| Escalation precision | 0.1597 | 0.1972 | 0.3814 |
| Auto-handle precision / recall | 0.9667 / 0.3031 | 0.9145 / 0.4427 | 0.7000 / 0.0875 |
| Guardrail pass rate | 1.0000 | 1.0000 | 1.0000 |
| PII violations | 0 | 0 | 0 |
| Resolution-act realization | 1.0000 | 1.0000 | 1.0000 |
| 280-character compliance | 1.0000 | 1.0000 | 1.0000 |

**Read the `random` column as the headline.** The `review` column is the hardest
382 rows in the corpus, so it is a floor, not an expectation, and quoting it
alone — as an earlier revision of this file did — understates the system by
roughly 29 accuracy points.

### How to read these numbers

- **The reference labels are weak supervision**, derived from historical agent
  replies by rule, not independent human judgement. They measure agreement with
  the supervision signal, not ground truth.
- **The labels describe the agent's reply, not the customer's intent**, and the
  model only ever sees the customer message. A damage complaint answered with
  refund language is labelled `REFUND_AND_RETURN`. Part of every label is
  unlearnable by construction, which is the dominant term in the remaining error.
- **The model is at the measured ceiling for this label set.** A plain TF-IDF +
  logistic-regression recipe reaches 0.8289/0.6414 on the same corpus
  (`scripts_feature_ab.py`), and the fitted pipeline reaches 0.8170/0.6142 — so
  the feature stack is not the bottleneck and the remaining gap is supervision,
  not capacity. More model will not help; better labels will.
- **Escalation recall is the safety-critical metric, and drift breaks its target.**
  On `random` it is 0.927 with a 0.822 lower bound; on `temporal` it falls to
  0.768 (lower bound 0.657), below the 0.85 target.
- **The drift shortfall is not fixable by re-deriving the threshold.** Sweeping
  every threshold over the temporal run's own risk scores
  (`scripts_recall_recoverability.py`) shows precision at a 0.85 recall floor is
  0.186, only **1.23× the base rate**, and it decays to 1.09× at a 0.95 floor. On
  same-period traffic the same sweep holds **1.42–1.56×** across the whole recall
  range. The score still reaches the floor, but only by escalating nearly
  everything — so the *ordering* has decayed, and the fix is retraining or new
  features. Anyone reaching for a threshold change here would be fixing the
  wrong thing.
- **The learned escalation score is modestly informative, not decisive.** Its
  lift over the base rate is 1.4–1.6× on same-period traffic and 1.1–1.3× under
  drift. The deterministic hard gates carry the safety property; the head mostly
  adds throughput.
- **The recall guarantee costs precision through overshoot.** The threshold is
  certified so that true recall is ≥0.85 *with 95% confidence*, which on ~250
  calibration positives means the observed recall runs to 0.95 at a precision of
  0.163 — against 0.186 at the 0.85 floor. If the operator will accept a
  point-estimate target rather than a confidence bound, the frontier below is the
  cheaper operating point. Expose it as policy, not as a fixed constant.
- **`review` is where the label noise shows.** Deployment accuracy 0.6152 against
  0.8174 on `random` — the active-learning sample is genuinely much harder, and
  the gap is the honest size of the "hard cases" effect.

### What moved the numbers

Each change is measured, not assumed. Several were fixes to *measurement*, and
the modelling "wins" they replaced were not real.

| Change | Effect |
|---|---|
| Nested blend admission with a paired-bootstrap gate | The prototype head held 0.6 of the final probability, fitted to calibration noise; now rejected outright. ECE 0.288 → 0.028 |
| Excluded model-generated labels from supervision | 40% of intent labels were produced by an earlier model trained on the same reply-derived labels, so they scored 0.898 out-of-fold against 0.643. Accuracy +0.024, precision +0.060 |
| Fit and serve the same confidence penalty | It fitted on untempered probabilities, selected rows by a quantity unavailable in production, and treated `0` as maximum rather than no penalty |
| Disabled class reweighting | Redundant once the deployment prior is restored: +0.010 accuracy, −0.05 ECE for −0.0001 macro-F1 |
| Widened the C search to 32 | The optimum sits at C=8–32; the grid stopped at 4 |
| Kept the natural class distribution | Balancing discarded ~80% of the dominant class's evidence |
| Split before capping/balancing | The calibration slice was itself flattened, so prior correction could not be selected |

On the `review` protocol — the only protocol measured before and after — accuracy
went 0.4690 → 0.5241, deployment accuracy 0.5277 → 0.6152, and ECE
0.2352 → 0.1848.

Diagnostics used to reach these numbers:

```powershell
.\.venv\Scripts\python.exe scripts_diagnose.py         # label noise, prior mismatch, majority floor
.\.venv\Scripts\python.exe scripts_cv_audit.py         # label circularity: reply-derived vs model-generated
.\.venv\Scripts\python.exe scripts_feature_ab.py       # is the feature stack the bottleneck? (no)
.\.venv\Scripts\python.exe scripts_preprocess_ab.py    # identifier normalisation (noise)
.\.venv\Scripts\python.exe scripts_ceiling.py          # what is learnable from customer text
.\.venv\Scripts\python.exe scripts_score.py --holdout random     # deployment-representative
.\.venv\Scripts\python.exe scripts_score.py --holdout temporal   # drift check
.\.venv\Scripts\python.exe scripts_score.py                      # hard-case review sample
.\.venv\Scripts\python.exe scripts_compare_reports.py A.json B.json   # paired comparison
.\.venv\Scripts\python.exe scripts_recall_recoverability.py models/x.json.rows.csv  # is a recall miss a stale threshold or decayed ranking?
```

The recoverability check exists because a recall shortfall has two possible
causes that call for opposite responses — a stale threshold, or a score that has
stopped ordering recent traffic — and they are distinguishable by sweeping
thresholds over the run's own risk scores. It reports which one you have.

`scripts_compare_reports.py` runs McNemar's exact test on the two runs' per-row
predictions when both are present, because configurations scored on the same rows
must be compared as paired data: most rows are easy and both systems get them
right, so they say nothing about which is better.

Tuning knobs (each participates in the cache fingerprint, so variants cache
separately):

| Knob | Default | Effect |
|---|---|---|
| `SUPPORT_INTENT_BALANCE_RATIO` | `0.0` (natural) | `>0` re-balances classes; measured to cost ~0.06 accuracy |
| `SUPPORT_INTENT_CAP_PER_CLASS` | `20000` | Lower values truncate the dominant class's evidence |
| `SUPPORT_INTENT_CLASS_WEIGHT` | `none` | `balanced` re-enables class reweighting |
| `SUPPORT_USE_PROBABILISTIC_LABELS` | `1` | Include model-generated intent labels in supervision (see below) |
| `SUPPORT_DISABLE_PIPELINE_CACHE` | unset | `1` forces a retrain instead of loading the cache |

### Evaluation protocols

The runner supports three holdout protocols, because a single number cannot
describe a model that has to survive drift. Reports record which protocol
produced them, and `scripts_compare_reports.py` refuses to compare across
protocols without saying so.

| Protocol | Command | What it measures |
|---|---|---|
| `review` | `--holdout review` | The 382 checked-in review rows. An **active-learning** sample (`active_learning_score` mean 0.767 over 0.52-0.93), i.e. the hardest rows in the corpus. A worst-case floor, not expected traffic. |
| `random` | `--holdout random` | A stratified random sample of the labelled corpus, excluded from training. Deployment-representative, and the default basis for reported numbers. |
| `temporal` | `--holdout temporal` | The newest 15% of conversations (`2017-12-02` to `2017-12-03` here). Train on the past, score the future — the only protocol that can fail for a reason other than sampling noise. |

### On supervision quality

Two facts about the labels bound what any model can score here, and both are
measured rather than assumed (`scripts_cv_audit.py`):

- **The labels describe the agent's reply, not the customer's intent.** They are
  extracted from historical replies by rule, so a damage complaint answered with
  refund language is labelled `REFUND_AND_RETURN`. The model only ever sees the
  customer message, so part of the label is unlearnable by construction. This is
  the dominant term in the remaining error and it is not fixable with modelling.
- **40% of the intent labels were produced by an earlier weak-label model**
  (5,616 of 14,067). Those labels score 0.898 out-of-fold macro-F1 against 0.643
  for reply-derived labels, i.e. they are largely reproducible from features the
  model already sees. Pooling them inflates any score computed over a label set
  that contains them — an earlier revision of this file reported 0.81
  cross-validated macro-F1 on that basis, against 0.64 for the honest subset.

## Security model

The service is designed to run on loopback for a single local operator. That
assumption is enforced rather than trusted:

| Control | Default | Override |
|---|---|---|
| Bind address | `127.0.0.1` | `--host` on `main.py serve` |
| Host header (DNS-rebinding guard) | `127.0.0.1`, `localhost`, `::1` | `SUPPORT_ALLOWED_HOSTS=host1,host2` |
| CORS | same-origin only (no CORS headers) | `SUPPORT_ALLOWED_ORIGINS=http://host:port,...` |
| Rate limit | 120 req/min per client, burst 30 | `SUPPORT_RATE_LIMIT_PER_MINUTE`, `SUPPORT_RATE_LIMIT_BURST` |
| Max request body | 64 KiB (`413` above) | `SUPPORT_MAX_BODY_BYTES` |
| Session IDs | `[A-Za-z0-9._:-]{1,80}` (`422` otherwise) | schema in `src/schemas.py` |
| Pipeline cache | SHA-256 manifest verified before `joblib.load` | `SUPPORT_DISABLE_PIPELINE_CACHE=1` to always retrain |
| API explorer (`/docs`) | enabled for local use | `SUPPORT_ENABLE_DOCS=0` to disable |
| Error bodies | generic `{"detail": "Internal server error."}` | server-side traceback only |

Response headers on every route: `Content-Security-Policy` (`default-src 'none'`,
`frame-ancestors 'none'`, inline styles/scripts only), `X-Content-Type-Options`,
`X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Permissions-Policy`,
`Cross-Origin-Opener-Policy`, `Cross-Origin-Resource-Policy`. API responses add
`Cache-Control: no-store`.

The serialized pipeline cache is the one place where untrusted bytes could lead to
code execution; `src/model_factory.py` therefore refuses to deserialize anything
without a matching `.sha256` manifest and treats mismatches as a cache miss.
Never point the service at a cache directory another user can write to.

Run the security regression suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_security.py -v
```

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
