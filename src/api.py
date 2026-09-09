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
from src.model_factory import build_learned_pipeline
from src.schemas import (
    ConfidenceRequest,
    ConfidenceResponse,
    EscalationResponse,
    HealthResponse,
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
    return HealthResponse(
        status="healthy",
        brand="@AmazonHelp",
        model_version="2.0.0-minilm-dual-head",
        corpus_size=pipeline_state["corpus_size"],
        intent_temperature=round(model.intent_temperature.temperature_, 4),
        escalation_temperature=round(model.escalation_temperature.temperature_, 4),
        escalation_threshold=round(model.escalation_threshold_, 4),
        generation_mode=pipeline_state["generation_mode"],
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
    started = time.time()
    model = pipeline_state["model"]
    classification = model.predict_with_confidence(req.text)
    learned_escalation = model.predict_escalation(req.text, classification["embedding"])
    escalator = pipeline_state["escalator"]
    decision = escalator.evaluate(
        req.text,
        classification["intent"],
        learned_escalation["risk_score"],
        is_uncertain=classification["is_uncertain"],
        learned_threshold=learned_escalation["threshold"],
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

    exemplars = pipeline_state["retriever"].retrieve_similar(
        req.text, intent_filter=classification["intent"], top_k=3
    )
    generator = pipeline_state["generator"]
    resolution_act = generator.select_act(exemplars, classification["intent"], decision.should_escalate)
    raw_reply = generator.generate(
        req.text,
        classification["intent"],
        exemplars,
        escalate_decision=decision.to_dict(),
    )
    guarded = pipeline_state["guard"].verify_and_sanitize(
        raw_reply, customer_text=req.text, should_escalate=decision.should_escalate
    )

    if session is not None:
        session.add_turn(
            req.text,
            classification["intent"],
            decision.should_escalate,
            decision.stated_reason,
            guarded.sanitized_reply,
        )

    latency_ms = round((time.time() - started) * 1000.0, 2)
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
    return TriageResponse(
        intent=classification["intent"],
        confidence=round(classification["confidence"], 4),
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
