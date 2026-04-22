import os
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv
import tiktoken

from src.scorer.relevance_scorer import MessageScore
from src.detector.landmark_detector import ProtectionLevel

load_dotenv()

THRESHOLD_HIGH = float(os.getenv("SELECTION_THRESHOLD_HIGH", "0.4"))
THRESHOLD_LOW = float(os.getenv("SELECTION_THRESHOLD_LOW", "0.2"))
PROTECTED_TAIL_SIZE = int(os.getenv("PROTECTED_TAIL_SIZE", "5"))


class SelectionAction(Enum):
    KEEP = "keep"
    COMPRESS = "compress"
    DROP = "drop"


@dataclass
class SelectionResult:
    """Selection decision for a single message."""
    index: int
    action: SelectionAction
    reason: str
    score: float


@dataclass
class SelectionSummary:
    """Full selection output for the pipeline."""
    selections: list[SelectionResult]
    kept_indices: list[int]
    compress_indices: list[int]
    dropped_indices: list[int]
    original_token_count: int
    kept_token_count: int
    estimated_reduction_pct: float


def _count_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    """Count tokens in a string using tiktoken."""
    return len(encoder.encode(text))


def _extract_text(message: dict) -> str:
    """Extract text content from a message."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    parts.append(block.get("name", ""))
                    parts.append(str(block.get("input", "")))
                elif block.get("type") == "tool_result":
                    parts.append(str(block.get("content", "")))
        return " ".join(parts)
    return str(content)


def _is_high_detail(text: str) -> bool:
    """
    Detect messages with specific details that compression would destroy.

    Matches: numbers/metrics, config values, code blocks, URLs, version numbers,
    enumerated steps, bullet lists with specifics, monetary amounts.
    """
    import re
    indicators = [
        r"\b\d+\.\d+\b",                    # decimal numbers (metrics, versions)
        r"\$\d+",                              # monetary amounts
        r"\b\d{2,}\b",                        # numbers with 2+ digits
        r"```",                                 # code blocks
        r"\b\d+\.\s+\w",                     # numbered steps (1. Do this)
        r"https?://",                           # URLs
        r"[A-Z_]{3,}=",                        # env vars / config (FOO=bar)
        r"\b(?:port|host|cpu|ram|gb|mb|ms)\b", # infrastructure terms
        r"-\s+\w",                             # bullet points with content
    ]
    matches = sum(1 for pat in indicators if re.search(pat, text, re.IGNORECASE))
    # Require at least 2 different indicators to avoid false positives
    return matches >= 2


def select_messages(
    messages: list[dict],
    scores: list[MessageScore],
    threshold_high: float = THRESHOLD_HIGH,
    threshold_low: float = THRESHOLD_LOW,
    tail_size: int = PROTECTED_TAIL_SIZE,
    min_coverage_pct: float = 0.2,
    query_type: str = "analytical",
) -> SelectionSummary:
    """
    Select which messages to keep, compress, or drop.

    Adaptive strategy based on query type:
    - Factual/landmark: more aggressive KEEP, less compression
    - Analytical: standard compression
    - Procedural: standard compression with recency bias (handled by scorer)

    Args:
        messages: Original conversation messages.
        scores: MessageScore list from relevance scorer (same order).
        threshold_high: Score above this → keep verbatim.
        threshold_low: Score below this → drop. Between low and high → compress.
        tail_size: Number of most recent messages to always keep.
        min_coverage_pct: Minimum fraction of messages to keep.
        query_type: One of 'factual', 'analytical', 'procedural', 'landmark'.

    Returns:
        SelectionSummary with per-message decisions and token counts.
    """
    if not messages:
        return SelectionSummary(
            selections=[],
            kept_indices=[],
            compress_indices=[],
            dropped_indices=[],
            original_token_count=0,
            kept_token_count=0,
            estimated_reduction_pct=0.0,
        )

    # --- Adaptive thresholds based on query type ---
    # Factual queries need precise answers — keep more, compress less
    if query_type in ("factual", "landmark"):
        threshold_high = max(threshold_high - 0.1, 0.15)
        threshold_low = max(threshold_low - 0.05, 0.1)
        min_coverage_pct = max(min_coverage_pct, 0.35)
    elif query_type == "analytical":
        # Analytical queries need broader coverage but still must hit
        # 40-60% reduction. Use standard thresholds with higher coverage floor.
        min_coverage_pct = max(min_coverage_pct, 0.25)
    elif query_type == "procedural":
        min_coverage_pct = max(min_coverage_pct, 0.25)

    encoder = tiktoken.get_encoding("cl100k_base")
    n = len(messages)
    tail_start = max(0, n - tail_size)

    selections = [None] * n

    # --- Pass 1: Apply priority rules ---

    for i, msg in enumerate(messages):
        score = scores[i]

        # Priority 0: First system message always kept
        if i == 0 and msg.get("role") == "system":
            selections[i] = SelectionResult(
                index=i, action=SelectionAction.KEEP,
                reason="system_message", score=score.final_score,
            )
            continue

        # Priority 1: PROTECTED landmarks → keep verbatim (NEVER compress)
        if score.protection == ProtectionLevel.PROTECTED:
            selections[i] = SelectionResult(
                index=i, action=SelectionAction.KEEP,
                reason="landmark_protected", score=score.final_score,
            )
            continue

        # Priority 2: CHAIN_PROTECTED tool chains → keep verbatim
        if score.protection == ProtectionLevel.CHAIN_PROTECTED:
            selections[i] = SelectionResult(
                index=i, action=SelectionAction.KEEP,
                reason="tool_chain_protected", score=score.final_score,
            )
            continue

        # Priority 3: Tail messages → keep verbatim
        if i >= tail_start:
            selections[i] = SelectionResult(
                index=i, action=SelectionAction.KEEP,
                reason="tail_protected", score=score.final_score,
            )
            continue

        # Priority 4 & 5: Score-based selection
        if score.final_score >= threshold_high:
            selections[i] = SelectionResult(
                index=i, action=SelectionAction.KEEP,
                reason="high_score", score=score.final_score,
            )
        elif score.final_score >= threshold_low:
            # For factual queries, promote mid-score to KEEP instead of COMPRESS
            if query_type in ("factual", "landmark") and score.final_score >= (threshold_high - 0.05):
                selections[i] = SelectionResult(
                    index=i, action=SelectionAction.KEEP,
                    reason="factual_promotion", score=score.final_score,
                )
            else:
                # Protect high-detail messages from compression —
                # numbers, configs, code, and steps get destroyed by summarisation
                text = _extract_text(msg)
                if _is_high_detail(text):
                    selections[i] = SelectionResult(
                        index=i, action=SelectionAction.KEEP,
                        reason="high_detail_protected", score=score.final_score,
                    )
                else:
                    selections[i] = SelectionResult(
                        index=i, action=SelectionAction.COMPRESS,
                        reason="mid_score", score=score.final_score,
                    )
        else:
            selections[i] = SelectionResult(
                index=i, action=SelectionAction.DROP,
                reason="low_score", score=score.final_score,
            )

    # --- Pass 2: Minimum coverage enforcement ---
    kept_count = sum(1 for s in selections if s.action in (SelectionAction.KEEP, SelectionAction.COMPRESS))
    min_keep = max(1, int(n * min_coverage_pct))

    if kept_count < min_keep:
        # Lower threshold: promote dropped messages by score until coverage met
        dropped = [s for s in selections if s.action == SelectionAction.DROP]
        dropped.sort(key=lambda s: s.score, reverse=True)

        for s in dropped:
            if kept_count >= min_keep:
                break
            # For factual queries, promote to KEEP; for others, COMPRESS
            promote_action = SelectionAction.KEEP if query_type in ("factual", "landmark") else SelectionAction.COMPRESS
            selections[s.index] = SelectionResult(
                index=s.index, action=promote_action,
                reason="coverage_promotion", score=s.score,
            )
            kept_count += 1

    # --- Compute token counts ---
    kept_indices = []
    compress_indices = []
    dropped_indices = []

    original_tokens = 0
    kept_tokens = 0

    for i, msg in enumerate(messages):
        text = _extract_text(msg)
        tokens = _count_tokens(text, encoder)
        original_tokens += tokens

        action = selections[i].action
        if action == SelectionAction.KEEP:
            kept_indices.append(i)
            kept_tokens += tokens
        elif action == SelectionAction.COMPRESS:
            compress_indices.append(i)
            # Estimate compressed tokens as ~30% of original
            kept_tokens += int(tokens * 0.3)
        else:
            dropped_indices.append(i)

    reduction_pct = 0.0
    if original_tokens > 0:
        reduction_pct = ((original_tokens - kept_tokens) / original_tokens) * 100

    return SelectionSummary(
        selections=selections,
        kept_indices=kept_indices,
        compress_indices=compress_indices,
        dropped_indices=dropped_indices,
        original_token_count=original_tokens,
        kept_token_count=kept_tokens,
        estimated_reduction_pct=round(reduction_pct, 1),
    )
