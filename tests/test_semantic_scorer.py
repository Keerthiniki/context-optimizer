"""Tests for Semantic Scorer — cosine similarity on MiniLM embeddings."""

import pytest
from src.scorer.semantic_scorer import SemanticScorer


# --- Semantic matching (the whole point) ---

def test_paraphrase_scores_higher_than_unrelated():
    """The key test: semantic similarity catches what BM25 misses."""
    messages = [
        {"role": "user", "content": "Let's go with PostgreSQL for our data store."},
        {"role": "assistant", "content": "The weather forecast looks sunny today."},
    ]
    scorer = SemanticScorer(messages)
    scores = scorer.score("What database did we pick?")

    # "database" and "PostgreSQL data store" are semantically close
    # "database" and "weather forecast" are not
    assert scores[0] > scores[1]


def test_conceptual_match():
    messages = [
        {"role": "user", "content": "We need to reduce server costs by optimizing resource usage."},
        {"role": "assistant", "content": "I like pizza with extra cheese."},
        {"role": "user", "content": "The infrastructure budget is too high this quarter."},
    ]
    scorer = SemanticScorer(messages)
    scores = scorer.score("How can we save money on cloud spending?")

    # Messages 0 and 2 are conceptually related to cost/infrastructure
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]


def test_exact_match_scores_high():
    messages = [
        {"role": "user", "content": "Deploy the application to production."},
        {"role": "assistant", "content": "Let's discuss the marketing strategy."},
    ]
    scorer = SemanticScorer(messages)
    scores = scorer.score("Deploy the application to production.")

    assert scores[0] == 1.0


# --- Normalization ---

def test_scores_normalized_zero_to_one():
    messages = [
        {"role": "user", "content": "Python is great for data science."},
        {"role": "assistant", "content": "JavaScript powers the web."},
        {"role": "user", "content": "Rust is fast and memory safe."},
    ]
    scorer = SemanticScorer(messages)
    scores = scorer.score("Which programming language should I learn?")

    for s in scores:
        assert 0.0 <= s <= 1.0
    assert max(scores) == 1.0


# --- Structured content ---

def test_structured_content():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "The database migration completed successfully."},
            ],
        },
        {"role": "user", "content": "Let's plan the team offsite."},
    ]
    scorer = SemanticScorer(messages)
    scores = scorer.score("How did the database migration go?")

    assert scores[0] > scores[1]


def test_tool_content():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "1", "name": "run_sql", "input": {"query": "SELECT count FROM users"}},
            ],
        },
        {"role": "user", "content": "The new logo design looks good."},
    ]
    scorer = SemanticScorer(messages)
    scores = scorer.score("database query user count")

    assert scores[0] > scores[1]


# --- Edge cases ---

def test_empty_messages():
    scorer = SemanticScorer([])
    scores = scorer.score("anything")
    assert scores == []


def test_empty_query():
    messages = [{"role": "user", "content": "Some content here."}]
    scorer = SemanticScorer(messages)
    scores = scorer.score("")
    assert scores == [0.0]


def test_empty_content_message():
    messages = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "The API returns user data."},
    ]
    scorer = SemanticScorer(messages)
    scores = scorer.score("user data API")

    # Empty message should score lower than relevant one
    assert scores[1] > scores[0]


def test_single_message():
    messages = [{"role": "user", "content": "Kubernetes cluster configuration."}]
    scorer = SemanticScorer(messages)
    scores = scorer.score("container orchestration setup")

    assert len(scores) == 1
    assert scores[0] == 1.0  # Only message, normalized to 1.0


# --- get_top_k ---

def test_get_top_k():
    messages = [
        {"role": "user", "content": "Set up the CI/CD pipeline with GitHub Actions."},
        {"role": "assistant", "content": "I had cereal for breakfast."},
        {"role": "user", "content": "The deployment workflow needs automated testing."},
        {"role": "assistant", "content": "My favourite colour is blue."},
        {"role": "user", "content": "Configure the build and release process."},
    ]
    scorer = SemanticScorer(messages)
    top = scorer.get_top_k("continuous integration deployment automation", k=3)

    assert len(top) == 3
    top_indices = {idx for idx, _ in top}
    # Messages 0, 2, 4 are about CI/CD — should dominate top 3
    assert top_indices == {0, 2, 4}
    # Scores should be descending
    assert top[0][1] >= top[1][1] >= top[2][1]
