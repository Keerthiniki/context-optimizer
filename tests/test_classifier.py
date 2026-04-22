"""Tests for Query Classifier — factual / analytical / procedural detection."""

import pytest
from src.classifier.query_classifier import classify_query


# --- Factual queries ---

@pytest.mark.parametrize("query", [
    "What did we decide about the database?",
    "Who is responsible for the deployment?",
    "When is the deadline?",
    "Which framework did we pick?",
    "What's the budget for Q3?",
    "Did we agree on the architecture?",
    "How much does the hosting cost?",
    "How many users do we have?",
    "Where is the config file?",
    "What was the timeline?",
])
def test_factual_queries(query):
    result = classify_query(query)
    assert result.query_type == "factual"
    assert result.confidence == "high"
    assert result.matched_pattern is not None


# --- Analytical queries ---

@pytest.mark.parametrize("query", [
    "Summarise our discussion about the backend.",
    "Compare React and Vue for our use case.",
    "What are the pros and cons of microservices?",
    "Give me an overview of what we covered.",
    "What are the key points from the meeting?",
    "Analyse the trade-offs we discussed.",
    "What did we discuss about deployment?",
    "Give me a recap of the architecture decisions.",
    "What are all the options we considered?",
    "Evaluate our current approach.",
])
def test_analytical_queries(query):
    result = classify_query(query)
    assert result.query_type == "analytical"
    assert result.confidence == "high"
    assert result.matched_pattern is not None


# --- Procedural queries ---

@pytest.mark.parametrize("query", [
    "How do we deploy to production?",
    "What are the steps to set up the database?",
    "Walk me through the onboarding process.",
    "How to configure the CI pipeline?",
    "Explain how the auth flow works.",
    "What's the process for code review?",
    "Guide me through the setup.",
    "Step-by-step instructions for deployment.",
    "What's the workflow for releasing a hotfix?",
    "How can I run the test suite?",
])
def test_procedural_queries(query):
    result = classify_query(query)
    assert result.query_type == "procedural"
    assert result.confidence == "high"
    assert result.matched_pattern is not None


# --- Fallback to analytical ---

@pytest.mark.parametrize("query", [
    "Tell me about the project.",
    "Thoughts?",
    "What's going on?",
    "Help me understand this.",
])
def test_fallback_to_analytical(query):
    result = classify_query(query)
    assert result.query_type == "analytical"
    assert result.confidence == "low"
    assert result.matched_pattern is None


# --- Edge cases ---

def test_empty_query():
    result = classify_query("")
    assert result.query_type == "analytical"
    assert result.confidence == "low"


def test_whitespace_query():
    result = classify_query("   ")
    assert result.query_type == "analytical"
    assert result.confidence == "low"


def test_case_insensitive():
    result = classify_query("WHAT DID WE DECIDE ABOUT THE API?")
    assert result.query_type == "factual"
    assert result.confidence == "high"


def test_result_has_all_fields():
    result = classify_query("Summarise the meeting.")
    assert hasattr(result, "query_type")
    assert hasattr(result, "confidence")
    assert hasattr(result, "matched_pattern")
