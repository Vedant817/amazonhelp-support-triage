"""Pydantic request and response schemas for the support pipeline."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


#: Session IDs are caller-supplied and used as in-memory dictionary keys, so they
#: are constrained to a short, log-safe character set.
SESSION_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,80}$"


class TriageRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = Field(
        None,
        max_length=80,
        pattern=SESSION_ID_PATTERN,
        description="Optional conversation key. Alphanumeric, dot, dash, underscore, colon.",
    )


class ConfidenceRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


class ConfidenceResponse(BaseModel):
    intent: str
    confidence: float
    threshold: float
    should_abstain: bool
    all_probabilities: Dict[str, float]


class EscalationResponse(BaseModel):
    should_escalate: bool
    stated_reason: str
    risk_score: float
    threshold: float
    triggers: List[str]


class RetrievalEvidence(BaseModel):
    reply_id: str
    resolution_act: str
    customer_text: str
    reply_text: str
    rrf_score: float
    dense_score: float
    bm25_score: float


class PipelineStageResponse(BaseModel):
    stage: str
    status: str
    duration_ms: float
    details: Dict[str, Any]


class TriageResponse(BaseModel):
    intent: str
    confidence: float
    all_probabilities: Dict[str, float]
    should_abstain: bool
    escalation: EscalationResponse
    resolution_act: str
    retrieval_evidence: List[RetrievalEvidence]
    generation_mode: str
    draft_reply: str
    guardrail_passed: bool
    guardrail_violations: List[str]
    character_count: int
    latency_ms: float
    pipeline_trace: List[PipelineStageResponse]


class HealthResponse(BaseModel):
    status: str
    brand: str
    model_version: str
    corpus_size: int
    historical_context_coverage: float
    trained_pipeline_cache_hit: bool
    intent_temperature: float
    escalation_temperature: float
    escalation_threshold: float
    high_precision_escalation_threshold: float
    intent_confidence_threshold: float
    target_escalation_recall: float
    escalation_calibration_recall: float
    escalation_recall_lower_bound: float
    escalation_recall_bound_target_met: bool
    escalation_calibration_positive_count: int
    generation_mode: str
    generation_backend: str
    generation_model_revision: Optional[str]
    intent_encoder: str
    intent_encoder_revision: str
    classification_features: str
    weak_intent_labels_pruned: int
    probabilistic_intent_label_count: int
    contrastive_training_pairs: int
    human_reason_training_count: int
    retrieval_method: str
    inference_device: str
    hosted_api_required: bool
