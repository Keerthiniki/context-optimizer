from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    """Request body for POST /optimize."""
    messages: list[dict] = Field(
        ...,
        min_length=1,
        description="Conversation messages, each with 'role' and 'content' keys.",
    )
    query: str = Field(
        ...,
        min_length=1,
        description="Current user query to optimise context for.",
    )
    token_budget: Optional[int] = Field(
        default=None,
        gt=0,
        description="Optional max token budget for optimised context.",
    )
    threshold_high: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Score threshold for KEEP (verbatim). Messages above this are kept.",
    )
    threshold_low: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Score threshold for COMPRESS. Messages between low and high are compressed.",
    )
    skip_compression: bool = Field(
        default=False,
        description="If True, skip Haiku compression (COMPRESS messages become DROP).",
    )


class BatchOptimizeRequest(BaseModel):
    """Request body for POST /optimize/batch — runs multiple optimisations."""
    requests: list[OptimizeRequest] = Field(
        ...,
        min_length=1,
        description="List of optimisation requests to process.",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ScoreBreakdown(BaseModel):
    """Score details for a single message."""
    index: int
    role: str
    action: str
    score: float
    content_preview: str = Field(
        default="",
        description="First 80 chars of message content for debugging.",
    )


class OptimizeResponse(BaseModel):
    """Response body for POST /optimize."""
    optimized_messages: list[dict] = Field(
        description="Optimised conversation thread ready for LLM consumption.",
    )
    metrics: "OptimizeMetrics"
    score_breakdown: list[ScoreBreakdown] = Field(
        default_factory=list,
        description="Per-message scoring details.",
    )


class OptimizeMetrics(BaseModel):
    """Metrics from a single optimisation run."""
    original_message_count: int
    optimized_message_count: int
    messages_kept: int
    messages_compressed: int
    messages_dropped: int
    summaries_injected: int
    token_count_original: int
    token_count_optimized: int
    token_reduction_percent: float
    landmarks_preserved: int
    tool_chains_preserved: int
    compression_cost_usd: float
    assembly_latency_ms: float
    thread_valid: bool


class BatchOptimizeResponse(BaseModel):
    """Response body for POST /optimize/batch."""
    results: list[OptimizeResponse]
    aggregate_metrics: "AggregateMetrics"


class AggregateMetrics(BaseModel):
    """Aggregate metrics across a batch of optimisation runs."""
    total_requests: int
    avg_token_reduction_percent: float
    total_compression_cost_usd: float
    avg_assembly_latency_ms: float
    all_threads_valid: bool


# Rebuild models to resolve forward references
OptimizeResponse.model_rebuild()
BatchOptimizeResponse.model_rebuild()
