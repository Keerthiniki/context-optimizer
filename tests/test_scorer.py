"""Tests for BM25 Scorer — keyword relevance scoring."""

import pytest
from src.scorer.bm25_scorer import BM25Scorer


# --- Basic scoring ---

def test_exact_keyword_match_scores_highest():
    messages = [
        {"role": "user", "content": "We should use PostgreSQL for the database."},
        {"role": "assistant", "content": "Sounds good, I agree."},
        {"role": "user", "content": "The weather is nice today."},
    ]
    scorer = BM25Scorer(messages)
    scores = scorer.score("PostgreSQL database")

    # Message 0 mentions both keywords — should score highest
    assert scores[0] == max(scores)
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_no_match_scores_zero():
    messages = [
        {"role": "user", "content": "Let's discuss the frontend framework."},
        {"role": "assistant", "content": "React or Vue?"},
    ]
    scorer = BM25Scorer(messages)
    scores = scorer.score("PostgreSQL database migration")

    assert all(s == 0.0 for s in scores)


def test_partial_match_scores_between():
    messages = [
        {"role": "user", "content": "We need a PostgreSQL database with Redis caching."},
        {"role": "assistant", "content": "PostgreSQL is a solid choice."},
        {"role": "user", "content": "Redis handles the session store."},
    ]
    scorer = BM25Scorer(messages)
    scores = scorer.score("PostgreSQL database")

    # Message 0 has both keywords, message 1 has one, message 2 has neither keyword
    assert scores[0] > scores[1]
    assert scores[1] > scores[2]


# --- Normalization ---

def test_scores_normalized_zero_to_one():
    messages = [
        {"role": "user", "content": "Deploy the application to production."},
        {"role": "assistant", "content": "I'll handle the deployment pipeline."},
        {"role": "user", "content": "Make sure to test first."},
    ]
    scorer = BM25Scorer(messages)
    scores = scorer.score("deployment production")

    for s in scores:
        assert 0.0 <= s <= 1.0
    # At least one score should be 1.0 (the max)
    assert max(scores) == 1.0


# --- Length normalization (BM25's advantage over raw counting) ---

def test_length_normalization():
    """Short message entirely about the topic should score higher than
    a long message that mentions it once among other things."""
    messages = [
        {"role": "user", "content": "PostgreSQL."},
        {"role": "assistant", "content": (
            "We discussed the frontend, backend, deployment, testing, "
            "monitoring, alerting, logging, caching, and also PostgreSQL "
            "among many other infrastructure components in our review."
        )},
    ]
    scorer = BM25Scorer(messages)
    scores = scorer.score("PostgreSQL")

    # Short focused message should score >= long unfocused one
    assert scores[0] >= scores[1]


# --- Structured content ---

def test_structured_content_scoring():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Here are the database migration results."},
            ],
        },
        {"role": "user", "content": "Thanks for the update on the frontend."},
    ]
    scorer = BM25Scorer(messages)
    scores = scorer.score("database migration")

    assert scores[0] > scores[1]


def test_tool_content_scoring():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "1", "name": "database_query", "input": {"sql": "SELECT * FROM users"}},
            ],
        },
        {"role": "user", "content": "What about the frontend styles?"},
    ]
    scorer = BM25Scorer(messages)
    scores = scorer.score("database query users")

    assert scores[0] > scores[1]


# --- Edge cases ---

def test_empty_messages():
    scorer = BM25Scorer([])
    scores = scorer.score("anything")
    assert scores == []


def test_empty_query():
    messages = [
        {"role": "user", "content": "Some content here."},
    ]
    scorer = BM25Scorer(messages)
    scores = scorer.score("")
    assert scores == [0.0]


def test_all_empty_content():
    messages = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
    ]
    scorer = BM25Scorer(messages)
    scores = scorer.score("test query")
    assert scores == [0.0, 0.0]


def test_single_message():
    messages = [
        {"role": "user", "content": "Deploy the app to AWS."},
    ]
    scorer = BM25Scorer(messages)
    scores = scorer.score("deploy AWS")
    assert len(scores) == 1
    assert scores[0] == 1.0


# --- get_top_k ---

def test_get_top_k():
    messages = [
        {"role": "user", "content": "The database needs indexing."},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "PostgreSQL index performance is critical."},
        {"role": "assistant", "content": "I agree about the database indexing strategy."},
        {"role": "user", "content": "Let's talk about frontend next."},
    ]
    scorer = BM25Scorer(messages)
    top = scorer.get_top_k("database indexing", k=2)

    assert len(top) == 2
    # Top results should be indices 0, 2, or 3 (the ones mentioning database/indexing)
    top_indices = {idx for idx, _ in top}
    assert top_indices.issubset({0, 2, 3})
    # Scores should be descending
    assert top[0][1] >= top[1][1]


def test_get_top_k_larger_than_corpus():
    messages = [
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "Hi there."},
    ]
    scorer = BM25Scorer(messages)
    top = scorer.get_top_k("hello", k=10)

    # Should return all messages, not crash
    assert len(top) == 2
