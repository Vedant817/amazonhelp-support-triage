# Hiver SDE Intern take-home submission

This repository contains a fully local `@AmazonHelp` support-agent prototype and its
evaluation evidence. It does not require an LLM API key.

Public repository: https://github.com/Vedant817/amazonhelp-support-triage

## Deliverable map

| Assignment requirement | Repository evidence |
|---|---|
| Runnable pipeline | [`README.md`](README.md), `python main.py benchmark`, `python main.py serve` |
| 150–250 hand-labelled examples | 200 rows in [`data/golden/golden_eval_set.csv`](data/golden/golden_eval_set.csv) |
| Sampling and labelling note | [`data/golden/SAMPLING_AND_LABELING_NOTE.md`](data/golden/SAMPLING_AND_LABELING_NOTE.md) |
| Frozen gold integrity | [`data/golden/MANIFEST.json`](data/golden/MANIFEST.json) and pre-run SHA-256 verification |
| Automated evaluation | [`scripts/reproduce_headline.py`](scripts/reproduce_headline.py) and [`results/headline_results.json`](results/headline_results.json) |
| LLM-as-judge rubric | [`src/llm_judge.py`](src/llm_judge.py) using revision-pinned local Qwen |
| Judge–human agreement evidence | [`results/local_judge_evaluation.json`](results/local_judge_evaluation.json) on 40 identical replies |
| Two baselines | Majority/always-escalate and TF-IDF/keyword/retrieve-replay in the benchmark |
| Failure analysis | “Top five failure modes” in [`REPORT.md`](REPORT.md) plus [`results/failure_examples.csv`](results/failure_examples.csv) |
| Misleading headline discussion | Exact section in [`REPORT.md`](REPORT.md) |
| One-more-week plan | Seven-day plan in [`REPORT.md`](REPORT.md) |
| Decision log | Fourteen decisions in [`REPORT.md`](REPORT.md) |

## Reproduce the headline result

From PowerShell on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py test
.\.venv\Scripts\python.exe main.py benchmark
```

The first benchmark downloads the revision-pinned MiniLM encoder. Later runs use a
fingerprinted local cache. The saved benchmark uses deterministic offline rewriting;
hosted inference and API credentials are not required.

## Verified headline

| Metric | Proposed system |
|---|---:|
| Intent accuracy | 0.795 |
| Intent macro-F1 | 0.781 |
| Escalation recall | 0.909 |
| Escalation precision | 0.571 |
| Exact reason match | 0.470 |
| Replies within 280 characters | 1.000 |
| No direct PII echo | 1.000 |

The local judge's apparent 4.263/5 mean is deliberately not a headline result. On 40
exact same replies, judge–human Spearman correlation is -0.161 and quadratic weighted
kappa is 0.000. The report treats this as a failed judge validation, not evidence of
reply quality.

## Five-minute demo path

1. Run `python main.py serve` and open `http://127.0.0.1:8000`.
2. Submit a routine tracking message and show intent probabilities, retrieval evidence,
   selected resolution act, routing decision, guarded draft, and execution trace.
3. Submit `My parcel says delivered but it is not here` and show the non-vetoable
   delivery-dispute escalation.
4. Submit `My account was hacked; my OTP is 123456` and show account-takeover routing
   plus public-sensitive-data handling.
5. Open [`results/headline_results.json`](results/headline_results.json), then explain the
   confidence intervals, false-positive escalation cost, and failed local-judge audit.

## Pre-send checklist

- [x] Public repository contains no model weights, raw 3M-row source dataset, or secrets.
- [x] Headline result reproduces from the documented command.
- [x] Gold IDs are excluded from training and retrieval and protected by a manifest.
- [x] Two baselines and row-level predictions are included.
- [x] Mandatory report sections and a 14-item decision log are included.
- [x] Local judge and same-reply agreement evidence are included without overstating them.
- [x] Unit/integration and adversarial safety suites pass.
- [x] Public GitHub URL is confirmed.
- [ ] Attach or link [`REPORT.md`](REPORT.md) in the email.

## Known evidence limits

This is a trustworthy prototype, not a production deployment. The 200-case set became a
development benchmark after repeated aggregate inspection. The system over-escalates,
reason exact match remains 0.470, the local judge failed agreement validation, and the
prepared 400-row training queue has not received two-person independent adjudication.
These limits are stated prominently rather than hidden behind the strongest metric.
