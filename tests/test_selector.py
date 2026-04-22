"""Tests for Message Selector — priority-based selection logic."""

import pytest
from src.selector.message_selector import (
    select_messages,
    SelectionAction,
    SelectionSummary,
)
from src.scorer.relevance_scorer import MessageScore
from src.detector.landmark_detector import ProtectionLevel


def _make_score(index, final_score, protection=ProtectionLevel.NONE):
    """Helper to create a MessageScore with minimal fields."""
    return MessageScore(
        index=index,
        keyword_score=0.0,
        semantic_score=0.0,
        recency_score=0.0,
        landmark_boost=1.0,
        final_score=final_score,
        protection=protection,
    )


# --- Priority 0: System message ---

def test_system_message_always_kept():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    scores = [_make_score(0, 0.01), _make_score(1, 0.01), _make_score(2, 0.01)]
    result = select_messages(messages, scores, tail_size=1)

    assert result.selections[0].action == SelectionAction.KEEP
    assert result.selections[0].reason == "system_message"


# --- Priority 1: Landmarks ---

def test_landmarks_always_kept():
    messages = [
        {"role": "user", "content": "We decided to use PostgreSQL."},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Filler message."},
    ]
    scores = [
        _make_score(0, 0.05, ProtectionLevel.PROTECTED),
        _make_score(1, 0.05),
        _make_score(2, 0.05),
    ]
    result = select_messages(messages, scores, tail_size=1)

    assert result.selections[0].action == SelectionAction.KEEP
    assert result.selections[0].reason == "landmark_protected"


# --- Priority 2: Tool chains ---

def test_tool_chains_always_kept():
    messages = [
        {"role": "user", "content": "Run query."},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "sql", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "done"}]},
        {"role": "assistant", "content": "Filler."},
    ]
    scores = [
        _make_score(0, 0.05),
        _make_score(1, 0.05, ProtectionLevel.CHAIN_PROTECTED),
        _make_score(2, 0.05, ProtectionLevel.CHAIN_PROTECTED),
        _make_score(3, 0.05),
    ]
    result = select_messages(messages, scores, tail_size=1)

    assert result.selections[1].action == SelectionAction.KEEP
    assert result.selections[1].reason == "tool_chain_protected"
    assert result.selections[2].action == SelectionAction.KEEP
    assert result.selections[2].reason == "tool_chain_protected"


# --- Priority 3: Tail messages ---

def test_tail_messages_kept():
    messages = [{"role": "user", "content": f"Message {i}"} for i in range(10)]
    scores = [_make_score(i, 0.01) for i in range(10)]
    result = select_messages(messages, scores, tail_size=3)

    # Last 3 messages always kept
    assert result.selections[7].action == SelectionAction.KEEP
    assert result.selections[7].reason == "tail_protected"
    assert result.selections[8].action == SelectionAction.KEEP
    assert result.selections[9].action == SelectionAction.KEEP


# --- Priority 4: High score ---

def test_high_score_kept():
    messages = [
        {"role": "user", "content": "Important technical discussion."},
        {"role": "assistant", "content": "Filler."},
    ]
    scores = [_make_score(0, 0.8), _make_score(1, 0.01)]
    result = select_messages(messages, scores, tail_size=0, threshold_high=0.4)

    assert result.selections[0].action == SelectionAction.KEEP
    assert result.selections[0].reason == "high_score"


# --- Priority 5: Mid score → compress ---

def test_mid_score_compressed():
    messages = [
        {"role": "user", "content": "Somewhat relevant discussion."},
        {"role": "assistant", "content": "Filler."},
    ]
    scores = [_make_score(0, 0.3), _make_score(1, 0.01)]
    result = select_messages(messages, scores, tail_size=0,
                             threshold_high=0.4, threshold_low=0.2,
                             query_type="procedural")

    assert result.selections[0].action == SelectionAction.COMPRESS
    assert result.selections[0].reason == "mid_score"


# --- Priority 6: Low score → drop ---

def test_low_score_dropped():
    """Low scoring messages in a large enough conversation get dropped."""
    messages = [{"role": "user", "content": f"Message {i}"} for i in range(10)]
    # First two score very low, rest score high
    scores = [_make_score(i, 0.01 if i < 2 else 0.8) for i in range(10)]
    result = select_messages(messages, scores, tail_size=0,
                             threshold_high=0.4, threshold_low=0.2,
                             min_coverage_pct=0.0)

    assert result.selections[0].action == SelectionAction.DROP
    assert result.selections[1].action == SelectionAction.DROP


# --- Minimum coverage enforcement ---

def test_coverage_promotion():
    """If too few messages selected, dropped messages get promoted."""
    messages = [{"role": "user", "content": f"Message {i}"} for i in range(20)]
    # All scores below threshold — everything would be dropped
    scores = [_make_score(i, 0.05 + i * 0.005) for i in range(20)]
    result = select_messages(messages, scores, tail_size=0,
                             threshold_high=0.4, threshold_low=0.2,
                             min_coverage_pct=0.3)

    # At least 30% should be kept or compressed
    active = len(result.kept_indices) + len(result.compress_indices)
    assert active >= 6  # 30% of 20


def test_coverage_promotes_highest_scored_first():
    """Promotion should pick highest-scored dropped messages first."""
    messages = [{"role": "user", "content": f"Message {i}"} for i in range(10)]
    scores = [_make_score(i, 0.01 * (i + 1)) for i in range(10)]
    result = select_messages(messages, scores, tail_size=0,
                             threshold_high=0.4, threshold_low=0.2,
                             min_coverage_pct=0.3)

    # Promoted messages should be the highest-scored dropped messages
    promoted = [s for s in result.selections if s.reason == "coverage_promotion"]
    if len(promoted) >= 2:
        promoted_scores = [p.score for p in promoted]
        non_promoted_dropped = [s for s in result.selections
                                if s.action == SelectionAction.DROP]
        # Every promoted message should score >= every remaining dropped one
        if non_promoted_dropped:
            assert min(promoted_scores) >= max(s.score for s in non_promoted_dropped)


# --- Token counting ---

def test_token_counts_populated():
    messages = [
        {"role": "user", "content": "This is a test message with some content."},
        {"role": "assistant", "content": "This is a response with different content."},
    ]
    scores = [_make_score(0, 0.8), _make_score(1, 0.05)]
    result = select_messages(messages, scores, tail_size=0)

    assert result.original_token_count > 0
    assert result.kept_token_count > 0
    assert result.estimated_reduction_pct >= 0.0


def test_reduction_percentage():
    """Dropping messages should produce measurable reduction."""
    messages = [{"role": "user", "content": f"Message content number {i} with enough words."} for i in range(10)]
    scores = [_make_score(i, 0.8 if i < 3 else 0.01) for i in range(10)]
    result = select_messages(messages, scores, tail_size=0)

    # Most messages dropped → significant reduction
    assert result.estimated_reduction_pct > 30.0


# --- Index lists ---

def test_index_lists_complete():
    """Every message index should appear in exactly one list."""
    messages = [{"role": "user", "content": f"Msg {i}"} for i in range(10)]
    scores = [_make_score(i, 0.1 * i) for i in range(10)]
    result = select_messages(messages, scores, tail_size=2)

    all_indices = set(result.kept_indices) | set(result.compress_indices) | set(result.dropped_indices)
    assert all_indices == set(range(10))

    # No overlaps
    assert not (set(result.kept_indices) & set(result.compress_indices))
    assert not (set(result.kept_indices) & set(result.dropped_indices))
    assert not (set(result.compress_indices) & set(result.dropped_indices))


# --- Edge cases ---

def test_empty_messages():
    result = select_messages([], [])
    assert result.selections == []
    assert result.original_token_count == 0


def test_single_message():
    messages = [{"role": "user", "content": "Only message."}]
    scores = [_make_score(0, 0.5)]
    result = select_messages(messages, scores, tail_size=1)

    assert len(result.selections) == 1
    assert result.selections[0].action == SelectionAction.KEEP


def test_all_landmarks():
    """If every message is a landmark, everything is kept."""
    messages = [{"role": "user", "content": f"Decision {i}"} for i in range(5)]
    scores = [_make_score(i, 0.5, ProtectionLevel.PROTECTED) for i in range(5)]
    result = select_messages(messages, scores, tail_size=0)

    assert len(result.kept_indices) == 5
    assert len(result.dropped_indices) == 0
