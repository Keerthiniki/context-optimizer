"""
Tests for FastAPI endpoints — /optimize, /optimize/batch, /health.

Uses mock compressor to avoid real Haiku API calls.
Tests the full pipeline integration through the API layer.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.compressor.compressor import (
    Compressor,
    CompressorOutput,
    CompressionResult,
    CompressionCluster,
)


client = TestClient(app)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

SIMPLE_CONVERSATION = [
    {"role": "user", "content": "Let's plan the database migration project."},
    {"role": "assistant", "content": "Sure, what database are we migrating from and to?"},
    {"role": "user", "content": "We're moving from MySQL to PostgreSQL."},
    {"role": "assistant", "content": "Good choice. PostgreSQL has better JSON support and extensibility."},
    {"role": "user", "content": "What about the timeline?"},
    {"role": "assistant", "content": "I'd estimate 3-4 weeks for a clean migration with testing."},
    {"role": "user", "content": "Ok sounds good, let's chat about something else for a bit."},
    {"role": "assistant", "content": "Sure, what's on your mind?"},
    {"role": "user", "content": "Did you see the game last night?"},
    {"role": "assistant", "content": "I don't watch sports, but I heard it was exciting!"},
    {"role": "user", "content": "Yeah it was great. Anyway, we decided to use PostgreSQL for the migration."},
    {"role": "assistant", "content": "Confirmed. PostgreSQL migration it is. I'll draft the plan."},
]


# ---------------------------------------------------------------------------
# Mock setup
# ---------------------------------------------------------------------------

def _mock_compress(annotated_messages):
    """Mock compressor that returns simple summaries without API calls."""
    from src.compressor.compressor import SelectionAction
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
            summary="Discussion about general topics.",
            from_cache=True,
        ))

    return CompressorOutput(
        results=results,
        total_clusters=len(results),
        cache_hits=len(results),
    )


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

class TestHealthCheck:

    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /optimize
# ---------------------------------------------------------------------------

class TestOptimizeEndpoint:

    @patch("src.api.routes._get_compressor")
    def test_basic_optimize(self, mock_get_compressor):
        """Basic optimisation request returns valid response."""
        mock_comp = MagicMock()
        mock_comp.compress.side_effect = _mock_compress
        mock_get_compressor.return_value = mock_comp

        response = client.post("/optimize", json={
            "messages": SIMPLE_CONVERSATION,
            "query": "What database did we decide to use?",
        })

        assert response.status_code == 200
        data = response.json()
        assert "optimized_messages" in data
        assert "metrics" in data
        assert "score_breakdown" in data

        metrics = data["metrics"]
        assert metrics["original_message_count"] == 12
        assert metrics["token_count_original"] > 0
        assert metrics["token_count_optimized"] > 0
        assert metrics["token_count_optimized"] <= metrics["token_count_original"]

    @patch("src.api.routes._get_compressor")
    def test_skip_compression(self, mock_get_compressor):
        """skip_compression=True skips Haiku calls entirely."""
        mock_comp = MagicMock()
        mock_get_compressor.return_value = mock_comp

        response = client.post("/optimize", json={
            "messages": SIMPLE_CONVERSATION,
            "query": "What's the migration timeline?",
            "skip_compression": True,
        })

        assert response.status_code == 200
        # Compressor should not be called
        mock_comp.compress.assert_not_called()

        data = response.json()
        assert data["metrics"]["compression_cost_usd"] == 0.0

    @patch("src.api.routes._get_compressor")
    def test_response_has_score_breakdown(self, mock_get_compressor):
        """Score breakdown includes per-message details."""
        mock_comp = MagicMock()
        mock_comp.compress.side_effect = _mock_compress
        mock_get_compressor.return_value = mock_comp

        response = client.post("/optimize", json={
            "messages": SIMPLE_CONVERSATION,
            "query": "What database did we decide to use?",
        })

        data = response.json()
        breakdown = data["score_breakdown"]
        assert len(breakdown) == 12  # one per input message
        for item in breakdown:
            assert "index" in item
            assert "role" in item
            assert "action" in item
            assert "score" in item
            assert item["action"] in ["keep", "compress", "drop"]

    @patch("src.api.routes._get_compressor")
    def test_custom_thresholds(self, mock_get_compressor):
        """Custom thresholds are respected."""
        mock_comp = MagicMock()
        mock_comp.compress.side_effect = _mock_compress
        mock_get_compressor.return_value = mock_comp

        response = client.post("/optimize", json={
            "messages": SIMPLE_CONVERSATION,
            "query": "What database?",
            "threshold_high": 0.6,
            "threshold_low": 0.3,
        })

        assert response.status_code == 200

    def test_empty_messages_rejected(self):
        """Empty messages list is rejected by Pydantic validation."""
        response = client.post("/optimize", json={
            "messages": [],
            "query": "test",
        })
        assert response.status_code == 422

    def test_missing_query_rejected(self):
        """Missing query field is rejected."""
        response = client.post("/optimize", json={
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert response.status_code == 422

    def test_empty_query_rejected(self):
        """Empty string query is rejected."""
        response = client.post("/optimize", json={
            "messages": [{"role": "user", "content": "hello"}],
            "query": "",
        })
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /optimize/batch
# ---------------------------------------------------------------------------

class TestBatchEndpoint:

    @patch("src.api.routes._get_compressor")
    def test_batch_optimize(self, mock_get_compressor):
        """Batch endpoint processes multiple requests."""
        mock_comp = MagicMock()
        mock_comp.compress.side_effect = _mock_compress
        mock_get_compressor.return_value = mock_comp

        response = client.post("/optimize/batch", json={
            "requests": [
                {
                    "messages": SIMPLE_CONVERSATION,
                    "query": "What database did we decide on?",
                },
                {
                    "messages": SIMPLE_CONVERSATION,
                    "query": "What's the migration timeline?",
                },
            ]
        })

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert "aggregate_metrics" in data

        agg = data["aggregate_metrics"]
        assert agg["total_requests"] == 2
        assert agg["avg_token_reduction_percent"] >= 0
        assert agg["total_compression_cost_usd"] >= 0

    def test_batch_empty_rejected(self):
        """Empty batch is rejected."""
        response = client.post("/optimize/batch", json={
            "requests": [],
        })
        assert response.status_code == 422
