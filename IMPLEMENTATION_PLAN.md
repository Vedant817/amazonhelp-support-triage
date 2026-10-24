# 8-Week Implementation Plan: AmazonHelp Support Triage System

## Current State Assessment

Measured under `scripts_score.py --holdout random` unless stated otherwise, i.e.
a 1,268-row deployment-representative holdout drawn from the labelled corpus and
excluded from training. `--holdout review` scores the 382-row checked-in review
set, which is an active-learning sample of the hardest rows in the corpus.

- **Core pipeline**: working (intent classification, escalation, retrieval, generation, guardrails)
- **Intent**: 0.817 accuracy, 0.614 macro-F1, 0.705 macro precision, ECE 0.039
- **Escalation**: 0.927 recall (one-sided 95% lower bound 0.822), 0.160 precision
- **Safety**: all seven hard gates verified; guardrail pass rate 1.000; zero PII violations
- **Test suite**: 285 tests, including an adversarial suite and a security middleware suite
- **Supervision**: reply-derived labels only (model-generated labels excluded by default,
  because they score 0.898 out-of-fold against 0.643 and are therefore largely circular)
- **Ceiling**: a plain TF-IDF + logistic-regression recipe reaches 0.829/0.641 on the same
  corpus, so the pipeline is at parity; the remaining error is label quality, not capacity

### Progress against this plan

| Week | Status | Notes |
|---|---|---|
| 1. Foundation | **Done** | Test suite, coverage config, dependency floors, reproducibility scripts |
| 1.6 Golden set | Superseded | Replaced by three holdout protocols (`review`/`random`/`temporal`), which are stronger than a static golden file because they cannot be tuned against |
| 2. Monitoring | **Partial** | Security headers, rate limiting, request IDs, health endpoint; no metrics export or alerting |
| 3. Human labelling | **Started, blocked on people** | The measurement apparatus is in place (`scripts_cv_audit.py` quantifies label circularity; the review-sample protocol scores hard cases). Producing better labels requires annotators, not code |
| 4. Contrastive fine-tuning | **Tried, disabled** | Corrupted the encoder on weak labels; left off with the gate documented |
| 5. Retrieval & generation | **Partial** | Retrieval in place; generation is template-grounded |
| 6. Safety hardening | **Done** | Unsafe pickle removed, cache manifest trust, CORS/host/body limits, PII leak fixed |
| 7. Deployment | **Partial** | Loopback service, dashboard, warm-up guard; no container or orchestration |
| 8. Documentation | **Partial** | README documents scores, protocols, knobs and the security model |

### What actually moved the numbers

Each change below is measured, not assumed. The order matters: several were
fixes to measurement rather than to modelling, and the modelling "wins" they
replaced were not real.

1. **Blend weights were fitted to calibration noise.** Auxiliary intent heads had
   their weights chosen as the best of up to 112 candidates on a single ~1,700-row
   split, ranked by macro-F1 (standard error ~0.01), while admission required a
   gain of only 0.001. The served model gave a nearest-centroid head over MiniLM
   embeddings 0.6 of the final probability. Fixed with a nested select/verify
   protocol plus a paired-bootstrap gate; the blend is now rejected outright.
2. **Model-generated labels were mixed into supervision.** 5,616 of 14,067 intent
   labels were produced by an earlier model trained on the same reply-derived
   supervision, so they are largely reproducible from features the model already
   sees. Excluded by default.
3. **The confidence penalty fitted and served different transforms.** It fitted on
   raw softmax outputs and served on temperature-scaled ones, selected rows by a
   quantity unavailable in production, and treated `0` as maximum rather than no
   penalty.
4. **Prior correction and class reweighting were double-compensating**, and the
   C grid stopped at 4 while the optimum sits at 8-32.
5. **Evaluation was protocol-blind.** The headline number came from the hardest
   382 rows in the corpus; reports now carry their protocol.

---

## WEEK 1: Foundation & Quality Infrastructure

### Sprint Goal: Establish bulletproof development infrastructure

| Task | Description | Acceptance Criteria |
|------|-------------|---------------------|
| 1.1 | Restore test suite | All unit/integration tests pass |
| 1.2 | Add pytest + coverage config | `pytest --cov=src --cov-fail-under=80` passes |
| 1.3 | Add pre-commit hooks | black, ruff, mypy, pytest on every commit |
| 1.4 | Add GitHub Actions CI | Tests run on PR, coverage reported |
| 1.5 | Add dependency scanning | safety/dependabot configured |
| 1.6 | Restore golden eval set | `data/golden/golden_eval_set.csv` + manifest |
| 1.6 | Add benchmark reproducibility test | Golden set benchmark runs in CI |

**Deliverable**: `make test` passes locally and in CI with >80% coverage

---

## WEEK 2: Monitoring & Observability

### Sprint Goal: Production-grade observability

| Task | Description | Acceptance Criteria |
|------|-------------|---------------------|
| 2.1 | Structured logging (structlog) | JSON logs with request_id, latency, stage timings |
| 2.2 | Prometheus metrics expansion | Histograms for each pipeline stage, error rates |
| 2.3 | Health check expansion | `/api/v1/health` includes model versions, data freshness, cache status |
| 2.4 | Request tracing | Full pipeline trace with correlation IDs |
| 2.5 | Alerting rules | PrometheusRule for latency > p99, error rate > 1%, escalation recall drift |
| 2.6 | Grafana dashboard | Pre-built dashboard for latency, throughput, accuracy, safety gates |
| 2.7 | Log aggregation ready | Structured logs compatible with Loki/ELK |

**Deliverable**: `make monitor` starts full observability stack locally

---

## WEEK 3: Human Labeling Workflow (Critical Path)

### Sprint Goal: Enable high-quality human annotations to break accuracy ceiling

| Task | Description | Acceptance Criteria |
|------|-------------|---------------------|
| 3.1 | Restore review UI (`review_app.py`) | Blind annotation interface at `/review` |
| 3.2 | Add reviewer management | Multiple reviewers, assignment, progress tracking |
| 3.3 | Restore adjudication script | `scripts/adjudicate_reviews.py` with Cohen's kappa |
| 3.4 | Add label quality metrics | Agreement rates, reviewer calibration, label distributions |
| 3.5 | Active learning integration | Uncertainty sampling for review queue prioritization |
| 3.6 | Review audit trail | Immutable logs of all labeling decisions |
| 3.7 | Export adjudicated labels | CSV with consensus + reviewer metadata |

**Deliverable**: `make review` launches full labeling workflow; 2 reviewers can complete 400 cases

---

## WEEK 4: Model Quality & Contrastive Fine-tuning

### Sprint Goal: Re-enable contrastive learning with quality gates

| Task | Description | Acceptance Criteria |
|------|-------------|---------------------|
| 4.1 | Fix contrastive fine-tuning | Use human labels (not weak labels) for contrastive pairs |
| 4.2 | Quality gates for fine-tuning | Only run if ≥48 human labels covering all 6 intents |
| 4.3 | Add fine-tuning monitoring | Log training loss, eval metrics, embedding drift |
| 4.4 | Embedding versioning | Cache fine-tuned models with git SHA + data fingerprint |
| 4.5 | A/B comparison framework | Compare frozen vs fine-tuned embeddings on holdout |
| 4.5 | Rollback mechanism | One-click revert to frozen embeddings |

**Deliverable**: `make finetune` runs contrastive training with human labels only

---

## WEEK 5: Retrieval & Generation Quality

### Sprint Goal: Improve retrieval precision and generation relevance

| Task | Description | Acceptance Criteria |
|------|-------------|---------------------|
| 5.1 | Cross-encoder reranking | Add `cross-encoder/ms-marco-MiniLM-L6-v2` for top-12 → top-3 |
| 5.2 | Retrieval ablation framework | Compare BM25, MiniLM, Hybrid, Cross-encoder on holdout |
| 5.3 | Generation quality metrics | Act realization rate, hallucination rate, style consistency |
| 5.4 | Local LLM evaluation | Qualify Qwen generation vs offline templates |
| 5.5 | Response diversity | Penalize near-verbatim historical replies |
| 5.6 | Retrieval caching | Cache embeddings + index for faster startup |

**Deliverable**: `make eval-retrieval` runs full retrieval/generation benchmark

---

## WEEK 6: Safety Hardening & Edge Cases

### Sprint Goal: Bulletproof safety for production

| Task | Description | Acceptance Criteria |
|------|-------------|---------------------|
| 6.1 | Adversarial test suite | 100+ adversarial cases: prompt injection, jailbreak, PII, legal threats |
| 6.2 | Safety gate unit tests | Each hard gate has ≥5 positive/negative test cases |
| 6.3 | PII detection hardening | Regex + NER for emails, phones, order IDs, credit cards |
| 6.4 | Financial commitment guard | Detect hallucinated refunds, cancellations, compensations |
| 6.5 | URL allowlist enforcement | Strict validation + auto-replacement |
| 6.6 | Character limit + truncation | 280 chars with smart truncation (preserve URLs, acts) |
| 6.7 | Regression test suite | Golden set + adversarial cases run on every PR |

**Deliverable**: `make test-safety` passes with 100% safety gate coverage

---

## WEEK 7: Deployment & Operations

### Sprint Goal: Production-ready deployment

| Task | Description | Acceptance Criteria |
|------|-------------|---------------------|
| 7.1 | Dockerfile (multi-stage) | <200MB image, non-root, health check |
| 7.2 | docker-compose (dev/staging/prod) | One-command full stack startup |
| 7.3 | Kubernetes manifests | Deployment, Service, HPA, PodDisruptionBudget |
| 7.4 | Secrets management | External secrets operator / Vault integration |
| 7.5 | Database migrations | Alembic for schema versioning (if adding persistent storage) |
| 7.6 | Blue-green deployment | Argo Rollouts / Flagger for zero-downtime |
| 7.7 | Runbook documentation | Incident response, rollback, scaling procedures |

**Deliverable**: `make deploy-staging` deploys to staging environment

---

## WEEK 8: Polish, Documentation & Launch Prep

### Sprint Goal: Launch-ready system with full documentation

| Task | Description | Acceptance Criteria |
|------|-------------|---------------------|
| 8.1 | API documentation (OpenAPI) | Complete spec at `/docs` with examples |
| 8.2 | Architecture decision records (ADRs) | 10+ ADRs for key design decisions |
| 8.3 | User guide | Setup, configuration, monitoring, troubleshooting |
| 8.4 | Contributor guide | Development setup, testing, PR process |
| 8.5 | Load testing | Locust script: 100 RPS sustained, p99 < 500ms |
| 8.6 | Chaos engineering | Pod kill, network partition, latency injection tests |
| 8.7 | Launch checklist | Security review, data privacy, compliance sign-off |

**Deliverable**: System ready for production launch with full documentation

---

## Commit Convention

```
feat(scope): brief description
fix(scope): brief description
docs(scope): brief description
test(scope): brief description
refactor(scope): brief description
ci(scope): brief description
chore(scope): brief description
```

Scopes: `api`, `model`, `retrieval`, `generation`, `safety`, `monitoring`, `deployment`, `tests`, `docs`, `ci`

---

## QA Engineer Personas (Simulated)

### QA-1: Security Engineer
- Focus: Prompt injection, PII leakage, financial commitment guards, URL allowlist
- Tools: Custom adversarial payloads, fuzzing, penetration testing mindset

### QA-2: Reliability Engineer
- Focus: Latency under load, error handling, cascade failures, graceful degradation
- Tools: Locust, chaos mesh, fault injection, latency injection

### QA-3: Data Quality Engineer
- Focus: Label quality, retrieval relevance, generation accuracy, drift detection
- Tools: Golden set evaluation, adversarial retrieval, generation quality metrics

---

## Testing Protocol Per Feature

1. **Unit Tests**: Every function/class has tests (≥90% coverage for new code)
2. **Integration Tests**: Full pipeline with mocked dependencies
3. **Contract Tests**: API schema validation, backward compatibility
4. **Golden Set Tests**: Benchmark on 200-case gold set (must not regress)
5. **Adversarial Tests**: 10+ adversarial cases per safety feature
6. **Load Tests**: 2x expected production load
6. **Chaos Tests**: Dependency failure, latency, partial degradation
7. **Smoke Tests**: Critical path verification post-deployment

---

## Definition of Done Per Feature

- [ ] All unit tests pass (>90% coverage)
- [ ] Integration tests pass
- [ ] Golden set benchmark runs (no regression on safety metrics)
- [ ] Adversarial tests pass (100% safety gate coverage)
- [ ] Code review approved (self-review + QA sign-off)
- [ ] Documentation updated (docstrings, API docs, runbooks)
- [ ] Monitoring/alerting configured for new feature
- [ ] Deployed to staging and smoke tested
- [ ] Performance budget met (p99 latency, memory, CPU)

---

## Weekly Rhythm

| Day | Activity |
|-----|----------|
| Mon | Sprint planning, task breakdown, commit previous week |
| Tue | Feature implementation (core logic) |
| Wed | Feature implementation (tests, edge cases) |
| Thu | QA testing (all 3 personas), bug fixes |
| Fri | Code review, documentation, commit, deploy to staging, retrospective |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Contrastive fine-tuning corrupts encoder | High | High | Quality gates, human-label-only, rollback mechanism |
| Weak label noise limits accuracy | Certain | High | Human labeling workflow (Week 3) |
| Retrieval latency under load | Medium | High | Caching, async retrieval, cross-encoder batching |
| PII detection gaps | Medium | Critical | Comprehensive test suite, NER + regex |
| Model drift in production | Medium | High | Drift detection, scheduled retraining, monitoring |
| Deployment complexity | Low | Medium | Docker Compose first, K8s incrementally |

---

## Success Metrics (End of 8 Weeks)

| Metric | Target |
|--------|--------|
| Escalation Recall | ≥ 95% (gold set) |
| Escalation Precision | ≥ 60% |
| Intent Macro-F1 | ≥ 75% (with human labels) |
| p99 Latency | < 500ms |
| Safety Gate Coverage | 100% (all 7 gates tested) |
| Test Coverage | > 85% |
| Adversarial Pass Rate | 100% |
| Deployment Time | < 5 min (staging) |
| MTTR | < 15 min |

---

*Plan created: 2026-10-23*
*Review weekly, adjust scope based on learnings*