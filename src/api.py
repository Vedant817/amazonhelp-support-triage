"""FastAPI service for calibrated dual-head triage and retrieve-then-rewrite."""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette import status as http_status
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.dashboard_html import DASHBOARD_HTML
from src.escalator import EscalationDecision, SafetyGatedEscalator
from src.generator import ProductionGroundedGenerator
from src.guardrails import ProductionPolicyGuard
from src.local_llm import QWEN_MODEL_ID, QWEN_REVISION
from src.model_factory import build_learned_pipeline
from src.schemas import (
    ConfidenceRequest,
    ConfidenceResponse,
    EscalationResponse,
    HealthResponse,
    PipelineStageResponse,
    RetrievalEvidence,
    TriageRequest,
    TriageResponse,
)
from src.security import (
    MAX_BODY_BYTES,
    TokenBucketRateLimiter,
    apply_security_headers,
    client_identifier,
    resolve_allowed_origins,
)
from src.session import CustomerSessionTracker


pipeline_state: Dict[str, Any] = {}
metrics_lock = threading.Lock()
metrics_counters = {
    "total_requests": 0,
    "escalated_requests": 0,
    "pii_blocked_count": 0,
    "total_latency_ms": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts = build_learned_pipeline(
        pairs_csv="data/processed/amazonhelp_context_pairs.csv",
        intent_cap_per_class=int(os.getenv("SUPPORT_INTENT_CAP_PER_CLASS", "1200")),
        retrieval_size=int(os.getenv("SUPPORT_RETRIEVAL_SIZE", "8000")),
        contrastive_pairs=int(os.getenv("SUPPORT_CONTRASTIVE_PAIRS", "0")),
    )
    generation_mode = os.getenv("SUPPORT_GENERATION_MODE", "offline").lower()
    pipeline_state.update(
        model=artifacts.model,
        retriever=artifacts.retriever,
        escalator=SafetyGatedEscalator(artifacts.model.escalation_threshold_),
        generator=ProductionGroundedGenerator(mode=generation_mode),
        guard=ProductionPolicyGuard(),
        session_tracker=CustomerSessionTracker(),
        corpus_size=len(artifacts.retriever.records),
        historical_context_coverage=artifacts.context_coverage,
        trained_pipeline_cache_hit=artifacts.cache_hit,
        probabilistic_intent_label_count=artifacts.probabilistic_intent_label_count,
        generation_mode=generation_mode,
    )
    yield
    pipeline_state.clear()


_docs_enabled = os.getenv("SUPPORT_ENABLE_DOCS", "1").strip().lower() not in {"0", "false", "no", "off"}

app = FastAPI(
    title="@AmazonHelp Support Agent API",
    description="Local MiniLM dual-head triage with hard safety gates and historical resolution-act grounding.",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
)

# Host-header validation blocks DNS-rebinding attacks, where a hostile domain
# resolves to 127.0.0.1 and the browser then treats this service as same-origin.
# Starlette normalises IPv6 hosts with brackets, so both forms are listed.
# Extra hosts can be added through SUPPORT_ALLOWED_HOSTS.
_allowed_hosts = ["127.0.0.1", "localhost", "[::1]", "::1", "testserver"]
_extra_hosts = [h.strip() for h in os.getenv("SUPPORT_ALLOWED_HOSTS", "").split(",") if h.strip()]
app.add_middleware(TrustedHostMiddleware, allowed_hosts=_allowed_hosts + _extra_hosts)

# CORS is same-origin by default. A browser page on another origin gets no CORS
# headers, so it cannot read responses or issue preflighted POSTs. An explicit
# allowlist can be opted into for a separate front-end dev server.
_allowed_origins = resolve_allowed_origins()
if _allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        max_age=600,
    )

_rate_limiter = TokenBucketRateLimiter()


@app.middleware("http")
async def security_middleware(request, call_next):
    """Enforce body-size caps, per-client throttling, and response hardening."""
    # 1. Reject oversized bodies before they are buffered or parsed.
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request body exceeds {MAX_BODY_BYTES} bytes."},
                )
        except ValueError:
            return JSONResponse(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                content={"detail": "Invalid Content-Length header."},
            )

    # 2. Throttle by socket peer (never by spoofable forwarded headers).
    if not _rate_limiter.allow(client_identifier(request)):
        return JSONResponse(
            status_code=http_status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Slow down and retry shortly."},
            headers={"Retry-After": "2"},
        )

    response = await call_next(request)

    # 3. Harden every response.
    apply_security_headers(response, api_route=request.url.path.startswith("/api/"))
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # pragma: no cover - logging path
    """Never leak stack traces or internal state to a caller."""
    logging.getLogger("amazonhelp.api").exception(
        "Unhandled error on %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


def _require_pipeline() -> None:
    """Fail fast with a clear 503 while the model is still warming up."""
    if "model" not in pipeline_state:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline is still loading. Retry shortly.",
        )


@app.get("/api/v1/health", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse, include_in_schema=False)
def health():
    _require_pipeline()
    model = pipeline_state["model"]
    threshold_selection = model.escalation_threshold_selection_
    encoder_device = getattr(getattr(model.encoder, "_model", None), "device", None)
    generation_mode = pipeline_state["generation_mode"]
    return HealthResponse(
        status="healthy",
        brand="@AmazonHelp",
        model_version="2.3.0-probabilistic-local-dual-head-context-ready",
        corpus_size=pipeline_state["corpus_size"],
        historical_context_coverage=round(pipeline_state["historical_context_coverage"], 4),
        trained_pipeline_cache_hit=pipeline_state["trained_pipeline_cache_hit"],
        intent_temperature=round(model.intent_temperature.temperature_, 4),
        escalation_temperature=round(model.escalation_temperature.temperature_, 4),
        escalation_threshold=round(model.escalation_threshold_, 4),
        high_precision_escalation_threshold=round(model.high_precision_escalation_threshold_, 4),
        intent_confidence_threshold=round(model.intent_confidence_threshold, 4),
        target_escalation_recall=round(model.target_escalation_recall, 4),
        escalation_calibration_recall=round(threshold_selection.observed_recall, 4),
        escalation_recall_lower_bound=round(threshold_selection.recall_lower_bound, 4),
        escalation_recall_bound_target_met=threshold_selection.target_met,
        escalation_calibration_positive_count=threshold_selection.positive_count,
        generation_mode=generation_mode,
        generation_backend=QWEN_MODEL_ID if generation_mode == "local" else "act-conditioned deterministic rewrite",
        generation_model_revision=QWEN_REVISION if generation_mode == "local" else None,
        intent_encoder=model.encoder.model_name,
        intent_encoder_revision=model.encoder.revision,
        classification_features=(
            "contrastive MiniLM + word TF-IDF + character TF-IDF"
            if model.encoder.contrastive_pairs_seen_ else
            "MiniLM + word TF-IDF + character TF-IDF"
        ),
        weak_intent_labels_pruned=model.intent_pruned_count_,
        probabilistic_intent_label_count=pipeline_state["probabilistic_intent_label_count"],
        contrastive_training_pairs=model.encoder.contrastive_pairs_seen_,
        human_reason_training_count=model.reason_training_count_,
        retrieval_method="BM25 + MiniLM reciprocal-rank fusion",
        inference_device=str(encoder_device or model.encoder.device or "cpu"),
        hosted_api_required=False,
    )


@app.post("/api/v1/confidence", response_model=ConfidenceResponse)
def confidence_lookup(req: ConfidenceRequest):
    _require_pipeline()
    model = pipeline_state["model"]
    result = model.predict_with_confidence(req.text)
    return ConfidenceResponse(
        intent=result["intent"],
        confidence=round(result["confidence"], 4),
        threshold=model.intent_confidence_threshold,
        should_abstain=result["is_uncertain"],
        all_probabilities={key: round(value, 4) for key, value in result["all_probabilities"].items()},
    )


@app.post("/api/v1/triage", response_model=TriageResponse)
def triage(req: TriageRequest):
    _require_pipeline()
    started = time.perf_counter()
    model = pipeline_state["model"]
    session = (
        pipeline_state["session_tracker"].get_or_create(req.session_id)
        if req.session_id else None
    )
    model_input = session.compose_model_input(req.text) if session else req.text
    prior_turns_used = min(len(session.turns), 2) if session else 0

    stage_started = time.perf_counter()
    classification = model.predict_with_confidence(req.text, model_input)
    classification_ms = (time.perf_counter() - stage_started) * 1000.0

    stage_started = time.perf_counter()
    current_embedding = classification["current_message_embedding"]
    learned_escalation = model.predict_escalation(req.text, current_embedding)
    reason_prediction = model.predict_reason(req.text, current_embedding)
    risk_scoring_ms = (time.perf_counter() - stage_started) * 1000.0

    stage_started = time.perf_counter()
    exemplars = pipeline_state["retriever"].retrieve_similar(
        model_input if classification["context_used"] else req.text,
        intent_filter=classification["intent"], top_k=12
    )
    generator = pipeline_state["generator"]
    evidence_act = generator.select_act(exemplars, classification["intent"], False)
    retrieval_ms = (time.perf_counter() - stage_started) * 1000.0

    stage_started = time.perf_counter()
    escalator = pipeline_state["escalator"]
    decision = escalator.evaluate(
        req.text,
        classification["intent"],
        learned_escalation["risk_score"],
        is_uncertain=classification["is_uncertain"],
        learned_threshold=learned_escalation["threshold"],
        high_precision_threshold=model.high_precision_escalation_threshold_,
        resolution_act=evidence_act,
        predicted_reason=(reason_prediction or {}).get("reason"),
    )

    if session is not None:
        friction = session.evaluate_multi_turn_friction(
            req.text, current_intent=classification["intent"]
        )
        if friction["override_escalate"]:
            decision = EscalationDecision(
                True,
                "HIGH_HOSTILITY_OR_LEGAL_RISK",
                max(float(friction["friction_score"]), decision.risk_score),
                decision.triggers + ["MULTI_TURN_FRICTION"],
            )
    escalation_ms = (time.perf_counter() - stage_started) * 1000.0

    resolution_act = generator.select_act(exemplars, classification["intent"], decision.should_escalate)

    stage_started = time.perf_counter()
    raw_reply, generation_metadata = generator.generate_with_metadata(
        req.text,
        classification["intent"],
        exemplars,
        escalate_decision=decision.to_dict(),
    )
    generation_ms = (time.perf_counter() - stage_started) * 1000.0

    stage_started = time.perf_counter()
    guarded = pipeline_state["guard"].verify_and_sanitize(
        raw_reply, customer_text=req.text, should_escalate=decision.should_escalate
    )
    guardrail_ms = (time.perf_counter() - stage_started) * 1000.0

    if session is not None:
        session.add_turn(
            req.text,
            classification["intent"],
            decision.should_escalate,
            decision.stated_reason,
            guarded.sanitized_reply,
        )

    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
    with metrics_lock:
        metrics_counters["total_requests"] += 1
        metrics_counters["escalated_requests"] += int(decision.should_escalate)
        metrics_counters["pii_blocked_count"] += int(any("PII" in item for item in guarded.violations))
        metrics_counters["total_latency_ms"] += latency_ms

    evidence = [
        RetrievalEvidence(
            reply_id=str(ex.get("reply_id", "")),
            resolution_act=str(ex.get("resolution_act", "GENERAL_SUPPORT")),
            customer_text=str(ex.get("customer_text", "")),
            reply_text=str(ex.get("reply_text", "")),
            rrf_score=float(ex.get("rrf_score", 0.0)),
            dense_score=float(ex.get("dense_score", 0.0)),
            bm25_score=float(ex.get("bm25_score", 0.0)),
        )
        for ex in exemplars[:3]
    ]
    hard_gate_triggered = any(str(trigger).startswith("HARD_GATE") for trigger in decision.triggers)
    pipeline_trace = [
        PipelineStageResponse(
            stage="Intent classification",
            status="abstained" if classification["is_uncertain"] else "complete",
            duration_ms=round(classification_ms, 2),
            details={
                "encoder": model.encoder.model_name,
                "features": (
                    "contrastive MiniLM + word/character TF-IDF"
                    if model.encoder.contrastive_pairs_seen_ else
                    "MiniLM + word/character TF-IDF"
                ),
                "supervision": "probabilistic weak-label aggregation + reviewed overrides",
                "human_reason_training_count": model.reason_training_count_,
                "weak_labels_pruned": model.intent_pruned_count_,
                "contrastive_training_pairs": model.encoder.contrastive_pairs_seen_,
                "predicted_intent": classification["intent"],
                "confidence": round(classification["confidence"], 4),
                "confidence_threshold": round(model.intent_confidence_threshold, 4),
                "resolution_method": classification["intent_resolution"],
                "causal_prior_turns_used": prior_turns_used,
                "context_selected_by_calibrated_gate": classification["context_used"],
                "current_message_confidence": round(classification["current_message_confidence"], 4),
                "context_confidence": (
                    round(classification["context_confidence"], 4)
                    if classification["context_confidence"] is not None else None
                ),
                "context_gate_confidence": round(model.context_gate_confidence_, 4),
                "context_gate_margin": round(model.context_gate_margin_, 4),
            },
        ),
        PipelineStageResponse(
            stage="Escalation risk scoring",
            status="complete",
            duration_ms=round(risk_scoring_ms, 2),
            details={
                "risk_score": round(learned_escalation["risk_score"], 4),
                "recall_threshold": round(learned_escalation["threshold"], 4),
                "high_precision_threshold": round(model.high_precision_escalation_threshold_, 4),
                "neighbor_blend_weight": round(model.escalation_knn_weight_, 4),
                "neighbor_count": model.escalation_knn_k_,
            },
        ),
        PipelineStageResponse(
            stage="Historical retrieval",
            status="complete" if evidence else "no_match",
            duration_ms=round(retrieval_ms, 2),
            details={
                "method": "BM25 + MiniLM RRF",
                "candidate_pool_size": len(exemplars),
                "neighbors_shown": len(evidence),
                "evidence_resolution_act": evidence_act,
                "top_rrf_score": evidence[0].rrf_score if evidence else None,
            },
        ),
        PipelineStageResponse(
            stage="Routing policy fusion",
            status="escalate" if decision.should_escalate else "auto_handle",
            duration_ms=round(escalation_ms, 2),
            details={
                "risk_score": round(decision.risk_score, 4),
                "recall_threshold": round(learned_escalation["threshold"], 4),
                "high_precision_threshold": round(model.high_precision_escalation_threshold_, 4),
                "retrieval_evidence_act": evidence_act,
                "hard_gate_triggered": hard_gate_triggered,
                "triggers": decision.triggers,
            },
        ),
        PipelineStageResponse(
            stage="Reply generation",
            status="complete",
            duration_ms=round(generation_ms, 2),
            details={
                "mode": pipeline_state["generation_mode"],
                "configured_backend": QWEN_MODEL_ID if pipeline_state["generation_mode"] == "local" else "act-conditioned deterministic rewrite",
                "actual_backend": generation_metadata["backend"],
                "fallback_reason": generation_metadata["fallback_reason"],
                "resolution_act": resolution_act,
            },
        ),
        PipelineStageResponse(
            stage="Pre-dispatch guardrails",
            status="passed" if guarded.is_valid else "adjusted",
            duration_ms=round(guardrail_ms, 2),
            details={
                "character_count": len(guarded.sanitized_reply),
                "violations": guarded.violations,
                "adjustments": guarded.adjustments_made,
            },
        ),
    ]
    return TriageResponse(
        intent=classification["intent"],
        confidence=round(classification["confidence"], 4),
        all_probabilities={
            key: round(value, 4) for key, value in classification["all_probabilities"].items()
        },
        should_abstain=classification["is_uncertain"],
        escalation=EscalationResponse(
            should_escalate=decision.should_escalate,
            stated_reason=decision.stated_reason,
            risk_score=round(decision.risk_score, 4),
            threshold=round(learned_escalation["threshold"], 4),
            triggers=decision.triggers,
        ),
        resolution_act=resolution_act,
        retrieval_evidence=evidence,
        generation_mode=pipeline_state["generation_mode"],
        draft_reply=guarded.sanitized_reply,
        guardrail_passed=guarded.is_valid,
        guardrail_violations=guarded.violations,
        character_count=len(guarded.sanitized_reply),
        latency_ms=latency_ms,
        pipeline_trace=pipeline_trace,
    )


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    with metrics_lock:
        requests = metrics_counters["total_requests"]
        escalations = metrics_counters["escalated_requests"]
        pii_blocks = metrics_counters["pii_blocked_count"]
        average_latency = metrics_counters["total_latency_ms"] / max(requests, 1)
    return (
        "# TYPE amazonhelp_requests_total counter\n"
        f"amazonhelp_requests_total {requests}\n"
        "# TYPE amazonhelp_escalations_total counter\n"
        f"amazonhelp_escalations_total {escalations}\n"
        "# TYPE amazonhelp_pii_blocks_total counter\n"
        f"amazonhelp_pii_blocks_total {pii_blocks}\n"
        "# TYPE amazonhelp_latency_ms_avg gauge\n"
        f"amazonhelp_latency_ms_avg {average_latency:.2f}\n"
    )
