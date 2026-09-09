"""
Production FastAPI REST Microservice for @AmazonHelp Support AI Pipeline.
Endpoints:
- GET  /: Interactive Web Dashboard UI.
- GET  /dashboard: Interactive Web Dashboard UI.
- POST /api/v1/triage: Full real-time support pipeline execution.
- POST /api/v1/conformal: Conformal prediction set lookup.
- GET  /api/v1/health: Health check and readiness probe.
- GET  /metrics: Prometheus monitoring metrics.
"""

import time
import os
import threading
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse

from src.schemas import (
    TriageRequest,
    TriageResponse,
    ConformalRequest,
    ConformalPredictionResponse,
    OODDetectionResponse,
    EscalationResponse,
    HealthResponse
)
from src.classifier import ProductionIntentClassifier, INTENT_TAXONOMY
from src.conformal import ConformalPredictor
from src.conformal_risk import ConformalRiskController
from src.escalator import ProductionEscalator
from src.retriever import HistoricalResolutionRetriever
from src.generator import ProductionGroundedGenerator
from src.guardrails import ProductionPolicyGuard
from src.session import CustomerSessionTracker
from src.ood_detector import EnergyOODDetector
from src.crag_retriever import CorrectiveRetriever
from src.dashboard_html import DASHBOARD_HTML

# Global pipeline singletons & thread-safe metric counters
pipeline_state: Dict[str, Any] = {}
metrics_lock = threading.Lock()
metrics_counters = {
    "total_requests": 0,
    "escalated_requests": 0,
    "pii_blocked_count": 0,
    "ood_count": 0,
    "total_latency_ms": 0.0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize pipeline
    train_csv = "data/processed/amazonhelp_english_pairs.csv"
    if not os.path.exists(train_csv):
        raise RuntimeError(f"Corpus file {train_csv} not found!")

    train_df = pd.read_csv(train_csv)
    # Exclude gold evaluation set if available to prevent leakage
    gold_csv = "data/golden/golden_eval_set.csv"
    if os.path.exists(gold_csv):
        gold_df = pd.read_csv(gold_csv)
        gold_ids = set(gold_df["tweet_id"].astype(str))
        clean_train_df = train_df[~train_df["inbound_id"].astype(str).isin(gold_ids)].reset_index(drop=True)
    else:
        clean_train_df = train_df

    train_sample = clean_train_df.sample(n=min(12000, len(clean_train_df)), random_state=42).reset_index(drop=True)

    X_train = train_sample["customer_text"].tolist()
    silver_labels = []
    for text in X_train:
        t_low = str(text).lower()
        if any(w in t_low for w in ["damaged", "broken", "wrong item", "defective", "missing item"]):
            silver_labels.append("DAMAGED_OR_WRONG_ITEM")
        elif any(w in t_low for w in ["login", "password", "otp", "2fa", "hacked", "account"]):
            silver_labels.append("ACCOUNT_AND_SECURITY")
        elif any(w in t_low for w in ["kindle", "fire tv", "alexa", "echo", "prime video"]):
            silver_labels.append("PRODUCT_AND_TECHNICAL")
        elif any(w in t_low for w in ["refund", "return", "returned", "charged twice", "billing"]):
            silver_labels.append("REFUND_AND_RETURN")
        elif any(w in t_low for w in ["track", "delivery", "carrier", "transit", "where is my", "package", "late", "shipping"]):
            silver_labels.append("ORDER_TRACKING_AND_DELAY")
        else:
            silver_labels.append("FEEDBACK_AND_GENERAL")

    cal_size = 1500
    X_fit = X_train[:-cal_size]
    y_fit = silver_labels[:-cal_size]
    X_cal = X_train[-cal_size:]
    y_cal = silver_labels[-cal_size:]

    classifier = ProductionIntentClassifier(confidence_threshold=0.32).fit(X_fit, y_fit)
    
    # Calibrate Conformal Predictor, OOD Detector & Conformal Risk Controller
    cal_probs = []
    cal_logits = []
    escalator_pre = ProductionEscalator(confidence_threshold=0.32)
    cal_risk_scores = []
    cal_escalate_truth = []

    for text, label in zip(X_cal, y_cal):
        res = classifier.predict_with_confidence(text)
        cal_probs.append(res["prob_vector"])
        cal_logits.append(res["logits"])

        # Determine pseudo-ground-truth high-risk escalation
        t_low = text.lower()
        is_true_high_risk = any(k in t_low for k in ["hacked", "stolen", "lawyer", "police", "charged twice", "never arrived", "broken"])
        esc_pre = escalator_pre.evaluate(text, intent=res["intent"], confidence=res["confidence"])
        cal_risk_scores.append(esc_pre.risk_score)
        cal_escalate_truth.append(is_true_high_risk)

    conformal = ConformalPredictor(alpha=0.05).calibrate(np.array(cal_probs), y_cal, classifier.classes_)
    ood_detector = EnergyOODDetector().calibrate(cal_logits)
    crc = ConformalRiskController(target_risk_alpha=0.05).calibrate(cal_risk_scores, cal_escalate_truth)

    retriever = HistoricalResolutionRetriever(top_k=3, rrf_k=20).build_index(train_sample)
    escalator = ProductionEscalator(confidence_threshold=0.32)
    generator = ProductionGroundedGenerator()
    guard = ProductionPolicyGuard()
    session_tracker = CustomerSessionTracker()
    crag = CorrectiveRetriever()

    pipeline_state["classifier"] = classifier
    pipeline_state["conformal"] = conformal
    pipeline_state["ood_detector"] = ood_detector
    pipeline_state["crc"] = crc
    pipeline_state["retriever"] = retriever
    pipeline_state["escalator"] = escalator
    pipeline_state["generator"] = generator
    pipeline_state["guard"] = guard
    pipeline_state["session_tracker"] = session_tracker
    pipeline_state["crag"] = crag
    pipeline_state["corpus_size"] = len(train_sample)

    yield
    pipeline_state.clear()


app = FastAPI(
    title="@AmazonHelp Support AI Agent API",
    description="Enterprise REST API for real-time customer support intent classification, conformal risk triage, and grounded response generation.",
    version="1.0.0",
    lifespan=lifespan
)

# Robust CORS configuration
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
    return HealthResponse(
        status="healthy",
        brand="@AmazonHelp",
        model_version="1.0.0",
        corpus_size=pipeline_state.get("corpus_size", 0),
        conformal_coverage_guarantee="95.0% Finite-Sample Split Conformal"
    )


@app.post("/api/v1/conformal", response_model=ConformalPredictionResponse)
def conformal_lookup(req: ConformalRequest):
    """Explicit endpoint for split conformal prediction set lookup."""
    clf: ProductionIntentClassifier = pipeline_state["classifier"]
    conf: ConformalPredictor = pipeline_state["conformal"]
    pred_res = clf.predict_with_confidence(req.text)
    conf_res = conf.predict_set(pred_res["prob_vector"])
    return ConformalPredictionResponse(
        prediction_set=conf_res["prediction_set"],
        set_size=conf_res["set_size"],
        is_uncertain=conf_res["is_uncertain"],
        confidence=round(conf_res["confidence"], 4),
        conformal_quantile=conf_res.get("conformal_quantile")
    )


@app.post("/api/v1/triage", response_model=TriageResponse)
def triage(req: TriageRequest):
    t0 = time.time()
    clf: ProductionIntentClassifier = pipeline_state["classifier"]
    conf: ConformalPredictor = pipeline_state["conformal"]
    ood: EnergyOODDetector = pipeline_state["ood_detector"]
    crc: ConformalRiskController = pipeline_state["crc"]
    ret: HistoricalResolutionRetriever = pipeline_state["retriever"]
    esc: ProductionEscalator = pipeline_state["escalator"]
    gen: ProductionGroundedGenerator = pipeline_state["generator"]
    guard: ProductionPolicyGuard = pipeline_state["guard"]
    crag: CorrectiveRetriever = pipeline_state["crag"]
    session_tracker: CustomerSessionTracker = pipeline_state["session_tracker"]

    # 1. Single-Pass Intent Classification & Logits
    pred_res = clf.predict_with_confidence(req.text)
    intent = pred_res["intent"]
    confidence = pred_res["confidence"]
    logits = pred_res["logits"]

    # 2. Split Conformal Prediction Set
    conf_res = conf.predict_set(pred_res["prob_vector"])

    # 3. Energy-based OOD Detection on Decision Logits (Single Pass)
    ood_res_dict = ood.score(logits)
    ood_res = OODDetectionResponse(
        is_ood=ood_res_dict["is_ood"],
        energy_score=ood_res_dict["energy_score"],
        energy_threshold=ood_res_dict["energy_threshold"],
        status=ood_res_dict["status"]
    )

    # 4. Multi-Turn Session Friction Tracking
    session_override = False
    session_reason = ""
    if req.session_id:
        sess = session_tracker.get_or_create(req.session_id)
        friction = sess.evaluate_multi_turn_friction(req.text)
        if friction["override_escalate"]:
            session_override = True
            session_reason = friction["reason"]

    # 5. Multi-Factor Escalation with Calibrated CRC Risk Threshold
    esc_dec = esc.evaluate(
        req.text,
        intent=intent,
        confidence=confidence,
        is_uncertain=conf_res["is_uncertain"] or ood_res.is_ood,
        conformal_risk_threshold=crc.calibrated_threshold_
    )
    if session_override:
        esc_dec.should_escalate = True
        esc_dec.stated_reason = session_reason
        esc_dec.triggers.append("MULTI_TURN_SESSION_TRIGGER")

    # 6. Hybrid Retrieval & CRAG Content-Token Verification
    exemplars = ret.retrieve_similar(req.text, intent_filter=intent, top_k=3)
    crag_res = crag.evaluate_and_correct(req.text, intent, exemplars)

    # 7. Grounded Generation with Verified KB Fallback & Pre-Dispatch Guardrail
    active_exemplars = crag_res["sanitized_exemplars"]
    raw_reply = gen.generate(
        req.text,
        intent,
        active_exemplars,
        escalate_decision=esc_dec.to_dict(),
        policy_guidance=crag_res.get("policy_guidance")
    )
    guard_res = guard.verify_and_sanitize(raw_reply, customer_text=req.text, should_escalate=esc_dec.should_escalate)

    # Record turn in session if tracking is active
    if req.session_id:
        sess.add_turn(
            customer_text=req.text,
            intent=intent,
            should_escalate=esc_dec.should_escalate,
            stated_reason=esc_dec.stated_reason,
            reply=guard_res.sanitized_reply
        )

    latency_ms = round((time.time() - t0) * 1000, 2)

    # Thread-Safe Telemetry Counters Update
    with metrics_lock:
        metrics_counters["total_requests"] += 1
        if esc_dec.should_escalate:
            metrics_counters["escalated_requests"] += 1
        if any("PII" in a or "Redacted" in a for a in guard_res.adjustments_made):
            metrics_counters["pii_blocked_count"] += 1
        if ood_res.is_ood:
            metrics_counters["ood_count"] += 1
        metrics_counters["total_latency_ms"] += latency_ms

    return TriageResponse(
        intent=intent,
        confidence=round(confidence, 4),
        conformal_prediction=ConformalPredictionResponse(
            prediction_set=conf_res["prediction_set"],
            set_size=conf_res["set_size"],
            is_uncertain=conf_res["is_uncertain"],
            confidence=round(conf_res["confidence"], 4),
            conformal_quantile=conf_res.get("conformal_quantile")
        ),
        ood_detection=ood_res,
        escalation=EscalationResponse(
            should_escalate=esc_dec.should_escalate,
            stated_reason=esc_dec.stated_reason,
            risk_score=esc_dec.risk_score,
            triggers=esc_dec.triggers
        ),
        crag_action=crag_res["action"],
        draft_reply=guard_res.sanitized_reply,
        guardrail_passed=guard_res.is_valid,
        character_count=len(guard_res.sanitized_reply),
        latency_ms=latency_ms
    )


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    with metrics_lock:
        req_count = metrics_counters["total_requests"]
        esc_count = metrics_counters["escalated_requests"]
        pii_count = metrics_counters["pii_blocked_count"]
        ood_count = metrics_counters["ood_count"]
        avg_lat = metrics_counters["total_latency_ms"] / max(req_count, 1)
        esc_rate = esc_count / max(req_count, 1)

    return (
        f"# HELP amazonhelp_requests_total Total triage requests processed\n"
        f"# TYPE amazonhelp_requests_total counter\n"
        f"amazonhelp_requests_total {req_count}\n\n"
        f"# HELP amazonhelp_escalations_total Total cases escalated to human agents\n"
        f"# TYPE amazonhelp_escalations_total counter\n"
        f"amazonhelp_escalations_total {esc_count}\n\n"
        f"# HELP amazonhelp_escalation_ratio Ratio of escalated cases\n"
        f"# TYPE amazonhelp_escalation_ratio gauge\n"
        f"amazonhelp_escalation_ratio {esc_rate:.4f}\n\n"
        f"# HELP amazonhelp_pii_blocks_total Total PII violations sanitized\n"
        f"# TYPE amazonhelp_pii_blocks_total counter\n"
        f"amazonhelp_pii_blocks_total {pii_count}\n\n"
        f"# HELP amazonhelp_ood_total Total out-of-distribution queries detected\n"
        f"# TYPE amazonhelp_ood_total counter\n"
        f"amazonhelp_ood_total {ood_count}\n\n"
        f"# HELP amazonhelp_latency_ms_avg Average request processing latency in ms\n"
        f"# TYPE amazonhelp_latency_ms_avg gauge\n"
        f"amazonhelp_latency_ms_avg {avg_lat:.2f}\n"
    )
