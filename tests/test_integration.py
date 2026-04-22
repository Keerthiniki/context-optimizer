"""
Integration test — full pipeline end-to-end.

Runs the complete optimize pipeline on a realistic conversation
with a mock compressor (no real Haiku calls). Verifies that all
components work together correctly.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.compressor.compressor import (
    CompressorOutput,
    CompressionResult,
    CompressionCluster,
    SelectionAction,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Realistic test conversation with landmarks
# ---------------------------------------------------------------------------

INTEGRATION_CONVERSATION = [
    {"role": "system", "content": "You are a helpful project planning assistant."},
    {"role": "user", "content": "Hey, let's start planning the Q3 product launch."},
    {"role": "assistant", "content": "Sure! What product are we launching?"},
    {"role": "user", "content": "It's the new AI search feature for our platform."},
    {"role": "assistant", "content": "Great. What's the target timeline?"},
    {"role": "user", "content": "We need it live by September 1st. That's the hard deadline."},
    {"role": "assistant", "content": "Understood. September 1st is the deadline. Let's work backwards from there."},
    {"role": "user", "content": "Yeah sounds good. By the way, did you watch the match last night?"},
    {"role": "assistant", "content": "I didn't catch it. How was it?"},
    {"role": "user", "content": "Amazing game, went to extra time. Anyway, back to planning."},
    {"role": "assistant", "content": "Haha nice. So for the AI search, we need to break this into phases."},
    {"role": "user", "content": "Right. What do you think about the tech stack?"},
    {"role": "assistant", "content": "I'd suggest Elasticsearch for indexing and a fine-tuned embedding model for semantic search."},
    {"role": "user", "content": "What about just using PostgreSQL full-text search?"},
    {"role": "assistant", "content": "PostgreSQL FTS works for simple keyword matching but won't handle semantic queries well. Elasticsearch gives us better relevance tuning."},
    {"role": "user", "content": "Ok I'm convinced. Let's go with Elasticsearch for the search backend."},
    {"role": "assistant", "content": "Agreed on Elasticsearch. I'll note that as a confirmed decision."},
    {"role": "user", "content": "Who's going to own the backend implementation?"},
    {"role": "assistant", "content": "I think Sarah from the platform team would be ideal. She has Elasticsearch experience."},
    {"role": "user", "content": "Good call. I'll handle the frontend integration myself."},
    {"role": "assistant", "content": "Sounds like a plan. So Sarah on backend, you on frontend."},
    {"role": "user", "content": "What about the coffee machine in the office? It's been broken for a week."},
    {"role": "assistant", "content": "Ha, I heard about that. Apparently facilities ordered a new one."},
    {"role": "user", "content": "About time. Ok so what's our testing strategy?"},
    {"role": "assistant", "content": "I'd recommend a two-week QA window before launch. August 15-31."},
    {"role": "user", "content": "That works. Action item: we need to write the test plan by August 1st."},
    {"role": "assistant", "content": "Noted. Test plan due by August 1st. I'll set up the tracking ticket."},
    {"role": "user", "content": "Perfect. Let's also make sure we have monitoring in place."},
    {"role": "assistant", "content": "Yes, we should set up Datadog dashboards for search latency and relevance metrics."},
    {"role": "user", "content": "Agreed. I think that covers the main planning items for now."},
    {"role": "assistant", "content": "Agreed. To summarise: Elasticsearch backend, Sarah owns backend, you own frontend, QA window Aug 15-31, test plan due Aug 1st, Datadog for monitoring. Deadline September 1st."},
]


def _mock_compress(annotated_messages):
    """Mock compressor for integration test."""
    clusters = []
    current_indices = []
    current_messages = []

    for am in annotated_messages:
        if am.action == SelectionAction.COMPRESS:
            current_indices.append(am.index)
            current_messages.append(am.message)
        else:
            if current_indices:
                clusters.append((list(current_indices), list(current_messages)))
                current_indices = []
                current_messages = []
    if current_indices:
        clusters.append((list(current_indices), list(current_messages)))

    results = []
    for indices, msgs in clusters:
        cluster = CompressionCluster(indices=indices, messages=msgs)
        results.append(CompressionResult(
            cluster=cluster,
            summary="Casual discussion about non-project topics.",
            from_cache=True,
        ))

    return CompressorOutput(
        results=results,
        total_clusters=len(results),
        cache_hits=len(results),
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end pipeline tests with realistic data."""

    @patch("src.api.routes._get_compressor")
    def test_factual_query_preserves_decision(self, mock_get_compressor):
        """Factual query about a decision preserves the landmark message."""
        mock_comp = MagicMock()
        mock_comp.compress.side_effect = _mock_compress
        mock_get_compressor.return_value = mock_comp

        response = client.post("/optimize", json={
            "messages": INTEGRATION_CONVERSATION,
            "query": "What search technology did we decide to use?",
        })

        assert response.status_code == 200
        data = response.json()

        # Decision about Elasticsearch should be preserved
        all_content = " ".join(
            m["content"] for m in data["optimized_messages"]
            if isinstance(m["content"], str)
        )
        assert "Elasticsearch" in all_content

        # Metrics should show reduction
        metrics = data["metrics"]
        assert metrics["token_reduction_percent"] > 0
        assert metrics["landmarks_preserved"] > 0

    @patch("src.api.routes._get_compressor")
    def test_analytical_query_broader_coverage(self, mock_get_compressor):
        """Analytical query keeps broader context."""
        mock_comp = MagicMock()
        mock_comp.compress.side_effect = _mock_compress
        mock_get_compressor.return_value = mock_comp

        response = client.post("/optimize", json={
            "messages": INTEGRATION_CONVERSATION,
            "query": "Summarise all the planning decisions we made.",
        })

        assert response.status_code == 200
        data = response.json()

        # Should preserve key decisions
        all_content = " ".join(
            m["content"] for m in data["optimized_messages"]
            if isinstance(m["content"], str)
        )
        assert "Elasticsearch" in all_content or "September" in all_content

    @patch("src.api.routes._get_compressor")
    def test_filler_messages_reduced(self, mock_get_compressor):
        """Low-value messages (sports, coffee) should be compressed or dropped."""
        mock_comp = MagicMock()
        mock_comp.compress.side_effect = _mock_compress
        mock_get_compressor.return_value = mock_comp

        response = client.post("/optimize", json={
            "messages": INTEGRATION_CONVERSATION,
            "query": "What's the project deadline?",
        })

        data = response.json()
        metrics = data["metrics"]

        # Some messages should be compressed or dropped
        assert metrics["messages_compressed"] + metrics["messages_dropped"] > 0

        # Should achieve meaningful reduction (lower bar for short conversations)
        assert metrics["token_reduction_percent"] > 5

    @patch("src.api.routes._get_compressor")
    def test_system_message_preserved(self, mock_get_compressor):
        """System message at position 0 is always kept."""
        mock_comp = MagicMock()
        mock_comp.compress.side_effect = _mock_compress
        mock_get_compressor.return_value = mock_comp

        response = client.post("/optimize", json={
            "messages": INTEGRATION_CONVERSATION,
            "query": "What did we decide?",
        })

        data = response.json()
        first_msg = data["optimized_messages"][0]
        assert first_msg["role"] == "system"
        assert "project planning" in first_msg["content"].lower()

    @patch("src.api.routes._get_compressor")
    def test_output_is_valid_thread(self, mock_get_compressor):
        """Assembled output passes thread validation."""
        mock_comp = MagicMock()
        mock_comp.compress.side_effect = _mock_compress
        mock_get_compressor.return_value = mock_comp

        response = client.post("/optimize", json={
            "messages": INTEGRATION_CONVERSATION,
            "query": "What are the next steps?",
        })

        data = response.json()
        messages = data["optimized_messages"]

        # Check role alternation (system exempt)
        for i in range(1, len(messages)):
            if messages[i]["role"] == "system":
                continue
            if messages[i - 1]["role"] == "system":
                continue
            assert messages[i]["role"] != messages[i - 1]["role"], (
                f"Consecutive {messages[i]['role']} at positions {i-1} and {i}"
            )

    @patch("src.api.routes._get_compressor")
    def test_score_breakdown_complete(self, mock_get_compressor):
        """Every input message has a score breakdown entry."""
        mock_comp = MagicMock()
        mock_comp.compress.side_effect = _mock_compress
        mock_get_compressor.return_value = mock_comp

        response = client.post("/optimize", json={
            "messages": INTEGRATION_CONVERSATION,
            "query": "What's the deadline?",
        })

        data = response.json()
        assert len(data["score_breakdown"]) == len(INTEGRATION_CONVERSATION)

        # Every entry has required fields
        for entry in data["score_breakdown"]:
            assert "index" in entry
            assert "score" in entry
            assert "action" in entry
            assert entry["action"] in ["keep", "compress", "drop"]

    @patch("src.api.routes._get_compressor")
    def test_token_counts_consistent(self, mock_get_compressor):
        """Optimised token count is less than original."""
        mock_comp = MagicMock()
        mock_comp.compress.side_effect = _mock_compress
        mock_get_compressor.return_value = mock_comp

        response = client.post("/optimize", json={
            "messages": INTEGRATION_CONVERSATION,
            "query": "Summarise the plan.",
        })

        data = response.json()
        metrics = data["metrics"]
        assert metrics["token_count_optimized"] < metrics["token_count_original"]
        assert metrics["token_count_optimized"] > 0
