"""FastAPI service for calibrated dual-head triage and retrieve-then-rewrite."""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse

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
        pairs_csv="data/processed/amazonhelp_english_pairs.csv",
        gold_csv="data/golden/golden_eval_set.csv",
        reviewed_csv="data/training/labels_for_human_review.csv",
        intent_cap_per_class=int(os.getenv("SUPPORT_INTENT_CAP_PER_CLASS", "1200")),
        retrieval_size=int(os.getenv("SUPPORT_RETRIEVAL_SIZE", "8000")),
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
        generation_mode=generation_mode,
    )
    yield
    pipeline_state.clear()


app = FastAPI(
    title="@AmazonHelp Support Agent API",
    description="Local MiniLM dual-head triage with hard safety gates and historical resolution-act grounding.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/api/v1/health", response_model=HealthResponse)
def health():
    model = pipeline_state["model"]
    encoder_device = getattr(getattr(model.encoder, "_model", None), "device", None)
    generation_mode = pipeline_state["generation_mode"]
    return HealthResponse(
        status="healthy",
        brand="@AmazonHelp",
        model_version="2.0.0-minilm-dual-head",
        corpus_size=pipeline_state["corpus_size"],
        intent_temperature=round(model.intent_temperature.temperature_, 4),
        escalation_temperature=round(model.escalation_temperature.temperature_, 4),
        escalation_threshold=round(model.escalation_threshold_, 4),
        high_precision_escalation_threshold=round(model.high_precision_escalation_threshold_, 4),
        intent_confidence_threshold=round(model.intent_confidence_threshold, 4),
        target_escalation_recall=round(model.target_escalation_recall, 4),
        generation_mode=generation_mode,
        generation_backend=QWEN_MODEL_ID if generation_mode == "local" else "act-conditioned deterministic rewrite",
        generation_model_revision=QWEN_REVISION if generation_mode == "local" else None,
        intent_encoder=model.encoder.model_name,
        intent_encoder_revision=model.encoder.revision,
        retrieval_method="BM25 + MiniLM reciprocal-rank fusion",
        inference_device=str(encoder_device or model.encoder.device or "cpu"),
        hosted_api_required=False,
    )


@app.post("/api/v1/confidence", response_model=ConfidenceResponse)
def confidence_lookup(req: ConfidenceRequest):
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
    started = time.perf_counter()
    model = pipeline_state["model"]

    stage_started = time.perf_counter()
    classification = model.predict_with_confidence(req.text)
    classification_ms = (time.perf_counter() - stage_started) * 1000.0

    stage_started = time.perf_counter()
    learned_escalation = model.predict_escalation(req.text, classification["embedding"])
    risk_scoring_ms = (time.perf_counter() - stage_started) * 1000.0

    stage_started = time.perf_counter()
    exemplars = pipeline_state["retriever"].retrieve_similar(
        req.text, intent_filter=classification["intent"], top_k=3
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
    )

    session = None
    if req.session_id:
        session = pipeline_state["session_tracker"].get_or_create(req.session_id)
        friction = session.evaluate_multi_turn_friction(req.text)
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
    raw_reply = generator.generate(
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
        for ex in exemplars
    ]
    hard_gate_triggered = any(str(trigger).startswith("HARD_GATE") for trigger in decision.triggers)
    pipeline_trace = [
        PipelineStageResponse(
            stage="Intent classification",
            status="abstained" if classification["is_uncertain"] else "complete",
            duration_ms=round(classification_ms, 2),
            details={
                "encoder": model.encoder.model_name,
                "predicted_intent": classification["intent"],
                "confidence": round(classification["confidence"], 4),
                "confidence_threshold": round(model.intent_confidence_threshold, 4),
                "resolution_method": classification["intent_resolution"],
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
            },
        ),
        PipelineStageResponse(
            stage="Historical retrieval",
            status="complete" if evidence else "no_match",
            duration_ms=round(retrieval_ms, 2),
            details={
                "method": "BM25 + MiniLM RRF",
                "neighbors_returned": len(evidence),
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
                "backend": QWEN_MODEL_ID if pipeline_state["generation_mode"] == "local" else "act-conditioned deterministic rewrite",
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
