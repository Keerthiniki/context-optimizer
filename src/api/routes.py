import time
from typing import Optional

import tiktoken
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException

from src.api.models import (
    OptimizeRequest,
    OptimizeResponse,
    OptimizeMetrics,
    ScoreBreakdown,
    BatchOptimizeRequest,
    BatchOptimizeResponse,
    AggregateMetrics,
)
from src.classifier.query_classifier import classify_query
from src.detector.landmark_detector import (
    detect_landmarks,
    get_protected_indices,
    ProtectionLevel,
)
from src.scorer.relevance_scorer import score_messages, get_weights_for_query_type
from src.selector.message_selector import select_messages, SelectionAction
from src.compressor.compressor import (
    Compressor,
    CompressorOutput,
    MessageWithAction,
    SelectionAction as CompressorSelectionAction,
)
from src.assembler.assembler import Assembler
from src.validator.thread_validator import ThreadValidator, ValidationError

load_dotenv()

router = APIRouter()

# ---------------------------------------------------------------------------
# Singleton components (loaded once, reused across requests)
# ---------------------------------------------------------------------------

_compressor: Optional[Compressor] = None
_assembler = Assembler()
_validator = ThreadValidator(strict=False)  # lenient — don't crash on warnings
_tokenizer = tiktoken.get_encoding("cl100k_base")


def _get_compressor() -> Compressor:
    """Lazy-load Compressor (initialises Anthropic client on first call)."""
    global _compressor
    if _compressor is None:
        _compressor = Compressor()
    return _compressor


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _count_tokens(messages: list[dict]) -> int:
    """Count tokens across all messages using tiktoken."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(_tokenizer.encode(content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "") or block.get("content", "")
                    if isinstance(text, str):
                        total += len(_tokenizer.encode(text))
        # Add role token overhead (~4 tokens per message for role + formatting)
        total += 4
    return total


def _content_preview(message: dict, max_len: int = 80) -> str:
    """Extract first max_len chars of message content for score breakdown."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content[:max_len]
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")[:max_len]
    return str(content)[:max_len]


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(request: OptimizeRequest) -> OptimizeResponse:
    """
    Execute the full optimisation pipeline for a single request.

    Steps:
    1. Classify query type (factual / analytical / procedural)
    2. Detect landmarks (decisions, commitments, tool chains)
    3. Score all messages (BM25 + semantic + recency + landmark)
    4. Select messages (KEEP / COMPRESS / DROP) — adaptive by query type
    5. Compress low-value clusters via Haiku
    6. Assemble valid conversation thread
    7. Validate structural integrity
    """
    pipeline_start = time.perf_counter()

    messages = request.messages
    query = request.query

    # --- Token count: original ---
    original_tokens = _count_tokens(messages)

    # --- Step 1: Classify query ---
    classification = classify_query(query)
    weights = get_weights_for_query_type(classification.query_type)

    # --- Step 2: Detect landmarks ---
    landmark_results = detect_landmarks(messages)
    protected_indices = get_protected_indices(landmark_results)

    # --- Step 3: Score messages ---
    scores = score_messages(messages, query, weights=weights)

  
# --- Step 4: Select messages (adaptive by query type) ---
    selection_summary = select_messages(
        messages=messages,
        scores=scores,
        threshold_high=request.threshold_high,
        threshold_low=request.threshold_low,
        query_type=classification.query_type,
    )

    # --- Step 4b: Pair preservation ---
    # When a message is KEEP, ensure its conversation partner (the next or
    # previous message in the user/assistant pair) is at least COMPRESS, not DROP.
    # This prevents orphaned questions without answers or answers without questions.
    from src.selector.message_selector import SelectionResult, _extract_text, _count_tokens as _sel_count_tokens

    for i, sel in enumerate(selection_summary.selections):
        if sel.action != SelectionAction.KEEP:
            continue
        role = messages[sel.index].get("role")

        # If user KEEP, check next message (assistant response)
        if role == "user" and i + 1 < len(selection_summary.selections):
            partner = selection_summary.selections[i + 1]
            if partner.action == SelectionAction.DROP:
                selection_summary.selections[i + 1] = SelectionResult(
                    index=partner.index,
                    action=SelectionAction.COMPRESS,
                    reason="pair_preservation",
                    score=partner.score,
                )

        # If assistant KEEP, check previous message (user question)
        if role == "assistant" and i - 1 >= 0:
            partner = selection_summary.selections[i - 1]
            if partner.action == SelectionAction.DROP:
                selection_summary.selections[i - 1] = SelectionResult(
                    index=partner.index,
                    action=SelectionAction.COMPRESS,
                    reason="pair_preservation",
                    score=partner.score,
                )

    # --- Step 4c: Quality guardrail ---
    # If too many messages dropped, promote highest-scored drops to KEEP.
    # Target: at least 40% of original tokens retained (max 60% reduction).
    _enc = tiktoken.get_encoding("cl100k_base")

    kept_tokens_est = sum(
        len(_enc.encode(_extract_text(messages[s.index])))
        for s in selection_summary.selections
        if s.action in (SelectionAction.KEEP, SelectionAction.COMPRESS)
    )
    orig_tokens_est = sum(
        len(_enc.encode(_extract_text(m)))
        for m in messages
    )
    # Quality guardrail: keep at least 35% of tokens (max 65% reduction).
    # Analytical/low-confidence get slightly higher floor to preserve breadth.
    if classification.query_type == "analytical" or classification.confidence == "low":
        min_retention = 0.38
    else:
        min_retention = 0.35

    if orig_tokens_est > 0 and (kept_tokens_est / orig_tokens_est) < min_retention:
        dropped_sels = [
            s for s in selection_summary.selections
            if s.action == SelectionAction.DROP
        ]
        dropped_sels.sort(key=lambda s: s.score, reverse=True)

        for s in dropped_sels:
            if kept_tokens_est / orig_tokens_est >= min_retention:
                break
            msg_tokens = len(_enc.encode(_extract_text(messages[s.index])))
            selection_summary.selections[s.index] = SelectionResult(
                index=s.index,
                action=SelectionAction.KEEP,
                reason="quality_guardrail",
                score=s.score,
            )
            kept_tokens_est += msg_tokens

    # --- Step 4d: Confidence-based analytical fallback ---
    # Compute confidence from selected messages: avg score + keyword coverage + landmarks
    kept_or_compressed = [
        s for s in selection_summary.selections
        if s.action in (SelectionAction.KEEP, SelectionAction.COMPRESS)
    ]

    if kept_or_compressed:
        avg_score = sum(s.score for s in kept_or_compressed) / len(kept_or_compressed)
        landmark_count = sum(
            1 for s in kept_or_compressed
            if s.index in protected_indices
        )
        confidence = (
            0.5 * avg_score
            + 0.3 * min(len(kept_or_compressed) / max(len(messages) * 0.3, 1), 1.0)
            + 0.2 * min(landmark_count / 3, 1.0)
        )
    else:
        confidence = 0.0

    # If analytical query with low confidence, expand context by promoting
    # top-scored dropped messages to KEEP (up to 10 extra messages)
    if classification.query_type == "analytical" and confidence < 0.4:
        dropped_sels = [
            s for s in selection_summary.selections
            if s.action == SelectionAction.DROP
        ]
        dropped_sels.sort(key=lambda s: s.score, reverse=True)

        expand_count = min(5, len(dropped_sels))
        for j in range(expand_count):
            s = dropped_sels[j]
            selection_summary.selections[s.index] = SelectionResult(
                index=s.index,
                action=SelectionAction.KEEP,
                reason="analytical_confidence_expansion",
                score=s.score,
            )

        # --- Build annotated messages for compressor ---
    annotated: list[MessageWithAction] = []
    score_breakdown: list[ScoreBreakdown] = []

    for sel in selection_summary.selections:
        # Map selector's SelectionAction to compressor's SelectionAction
        action_map = {
            SelectionAction.KEEP: CompressorSelectionAction.KEEP,
            SelectionAction.COMPRESS: CompressorSelectionAction.COMPRESS,
            SelectionAction.DROP: CompressorSelectionAction.DROP,
        }
        action = action_map[sel.action]

        # If compression is skipped, demote COMPRESS to DROP
        if request.skip_compression and action == CompressorSelectionAction.COMPRESS:
            action = CompressorSelectionAction.DROP

        annotated.append(MessageWithAction(
            index=sel.index,
            message=messages[sel.index],
            action=action,
            score=sel.score,
        ))

        score_breakdown.append(ScoreBreakdown(
            index=sel.index,
            role=messages[sel.index].get("role", "unknown"),
            action=action.value,
            score=round(sel.score, 4),
            content_preview=_content_preview(messages[sel.index]),
        ))

    # --- Step 5: Compress ---
    compressor = _get_compressor()
    if request.skip_compression:
        compressor_output = CompressorOutput()
    else:
        compressor_output = compressor.compress(annotated)

    # --- Step 6: Assemble ---
    assembler_output = _assembler.assemble(annotated, compressor_output)

    # --- Step 7: Validate ---
    validation_result = _validator.validate(assembler_output.raw_messages)

    # --- Token count: optimised ---
    optimized_tokens = _count_tokens(assembler_output.raw_messages)

    # --- Calculate metrics ---
    pipeline_ms = (time.perf_counter() - pipeline_start) * 1000

    reduction_pct = 0.0
    if original_tokens > 0:
        reduction_pct = round(
            (1 - optimized_tokens / original_tokens) * 100, 2
        )

    # Count preserved landmarks and tool chains
    landmarks_preserved = sum(
        1 for sel in selection_summary.selections
        if sel.index in protected_indices
        and sel.action == SelectionAction.KEEP
        and landmark_results[sel.index].protection == ProtectionLevel.PROTECTED
    )
    tool_chains_preserved = sum(
        1 for sel in selection_summary.selections
        if sel.index in protected_indices
        and sel.action == SelectionAction.KEEP
        and landmark_results[sel.index].protection == ProtectionLevel.CHAIN_PROTECTED
    )

    metrics = OptimizeMetrics(
        original_message_count=len(messages),
        optimized_message_count=len(assembler_output.raw_messages),
        messages_kept=assembler_output.total_kept,
        messages_compressed=assembler_output.total_compressed,
        messages_dropped=assembler_output.total_dropped,
        summaries_injected=assembler_output.summaries_injected,
        token_count_original=original_tokens,
        token_count_optimized=optimized_tokens,
        token_reduction_percent=reduction_pct,
        landmarks_preserved=landmarks_preserved,
        tool_chains_preserved=tool_chains_preserved,
        compression_cost_usd=round(compressor_output.total_cost_usd, 6),
        assembly_latency_ms=round(pipeline_ms, 2),
        thread_valid=validation_result.valid,
    )

    return OptimizeResponse(
        optimized_messages=assembler_output.raw_messages,
        metrics=metrics,
        score_breakdown=score_breakdown,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest):
    """
    Optimise a conversation context for a given query.

    Takes a multi-turn conversation and current query, returns an optimised
    context that is 40-60% smaller while preserving answer quality.
    """
    try:
        return run_pipeline(request)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@router.post("/optimize/batch", response_model=BatchOptimizeResponse)
async def optimize_batch(request: BatchOptimizeRequest):
    """
    Batch optimisation for evaluation framework.

    Processes multiple conversations and returns individual results
    plus aggregate metrics.
    """
    results = []
    for req in request.requests:
        try:
            result = run_pipeline(req)
            results.append(result)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline error on request {len(results)}: {str(e)}",
            )

    # Aggregate metrics
    total = len(results)
    avg_reduction = sum(r.metrics.token_reduction_percent for r in results) / total if total else 0
    total_cost = sum(r.metrics.compression_cost_usd for r in results)
    avg_latency = sum(r.metrics.assembly_latency_ms for r in results) / total if total else 0
    all_valid = all(r.metrics.thread_valid for r in results)

    return BatchOptimizeResponse(
        results=results,
        aggregate_metrics=AggregateMetrics(
            total_requests=total,
            avg_token_reduction_percent=round(avg_reduction, 2),
            total_compression_cost_usd=round(total_cost, 6),
            avg_assembly_latency_ms=round(avg_latency, 2),
            all_threads_valid=all_valid,
        ),
    )
