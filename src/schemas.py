"""Pydantic request and response schemas for the support pipeline."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None


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


class TriageResponse(BaseModel):
    intent: str
    confidence: float
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


class HealthResponse(BaseModel):
    status: str
    brand: str
    model_version: str
    corpus_size: int
    intent_temperature: float
    escalation_temperature: float
    escalation_threshold: float
    generation_mode: str
