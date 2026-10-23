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
