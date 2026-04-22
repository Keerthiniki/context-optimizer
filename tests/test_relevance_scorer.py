"""Tests for Relevance Scorer — combined scoring with all four signals."""

import pytest
from src.scorer.relevance_scorer import (
    score_messages,
    get_weights_for_query_type,
    ScoringWeights,
    MessageScore,
    LANDMARK_BOOST,
)
from src.detector.landmark_detector import ProtectionLevel


# --- Basic scoring ---

def test_relevant_message_scores_higher():
    messages = [
        {"role": "user", "content": "We decided to use PostgreSQL for the database."},
        {"role": "assistant", "content": "Sounds good."},
        {"role": "user", "content": "The weather is nice today."},
    ]
    scores = score_messages(messages, "What database did we choose?")

    # Message 0 is both keyword-relevant and semantically relevant
    assert scores[0].final_score > scores[2].final_score


def test_recent_messages_get_recency_boost():
    messages = [
        {"role": "user", "content": "Let's discuss the API design."},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "Let's discuss the API design."},
    ]
    scores = score_messages(messages, "API design")

    # Same content, but message 9 is more recent — should score higher
    assert scores[9].final_score > scores[0].final_score
    assert scores[9].recency_score > scores[0].recency_score


def test_landmark_gets_boosted():
    messages = [
        {"role": "user", "content": "We decided to use Kubernetes for orchestration."},
        {"role": "assistant", "content": "Kubernetes is a good choice for containers."},
    ]
    scores = score_messages(messages, "container orchestration")

    # Message 0 has a decision landmark — should get boost
    assert scores[0].protection == ProtectionLevel.PROTECTED
    assert scores[0].landmark_boost == LANDMARK_BOOST
    assert scores[1].landmark_boost == 1.0


# --- Score structure ---

def test_returns_correct_count():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Bye"},
    ]
    scores = score_messages(messages, "greeting")
    assert len(scores) == 3


def test_score_breakdown_populated():
    messages = [{"role": "user", "content": "Deploy to production."}]
    scores = score_messages(messages, "deploy production")

    assert isinstance(scores[0], MessageScore)
    assert scores[0].index == 0
    assert scores[0].keyword_score >= 0.0
    assert scores[0].semantic_score >= 0.0
    assert scores[0].recency_score >= 0.0
    assert scores[0].final_score >= 0.0


def test_scores_clamped_to_one():
    """Landmark boost can push raw score above 1.0 — should be clamped."""
    messages = [
        {"role": "user", "content": "We decided to use the exact deployment strategy."},
    ]
    # Use a query that matches perfectly + landmark boost
    scores = score_messages(messages, "We decided to use the exact deployment strategy")

    assert scores[0].final_score <= 1.0


# --- Weights ---

def test_custom_weights():
    messages = [
        {"role": "user", "content": "PostgreSQL database setup."},
        {"role": "assistant", "content": "Something unrelated entirely."},
    ]
    # Heavy keyword weight
    keyword_heavy = ScoringWeights(keyword=0.7, semantic=0.1, recency=0.1, landmark=0.1)
    scores = score_messages(messages, "PostgreSQL database", weights=keyword_heavy)

    # Keyword-heavy weights should still rank the relevant message first
    assert scores[0].final_score > scores[1].final_score


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        ScoringWeights(keyword=0.5, semantic=0.5, recency=0.5, landmark=0.5)


def test_get_weights_for_query_type():
    factual = get_weights_for_query_type("factual")
    assert factual.keyword == 0.3
    assert factual.landmark == 0.3

    analytical = get_weights_for_query_type("analytical")
    assert analytical.semantic == 0.3

    procedural = get_weights_for_query_type("procedural")
    assert procedural.recency == 0.3

    # Unknown type returns equal weights
    default = get_weights_for_query_type("unknown")
    assert default.keyword == 0.25


# --- Tool chain protection ---

def test_tool_chain_protected():
    messages = [
        {"role": "user", "content": "Run the database query."},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "1", "name": "sql_query", "input": {"sql": "SELECT *"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": "5 rows returned."},
            ],
        },
    ]
    scores = score_messages(messages, "database query results")

    assert scores[1].protection == ProtectionLevel.CHAIN_PROTECTED
    assert scores[2].protection == ProtectionLevel.CHAIN_PROTECTED


# --- Edge cases ---

def test_empty_messages():
    scores = score_messages([], "anything")
    assert scores == []


def test_empty_query():
    messages = [{"role": "user", "content": "Some content."}]
    scores = score_messages(messages, "")

    # Should still return scores (recency still applies)
    assert len(scores) == 1
    assert scores[0].final_score >= 0.0
