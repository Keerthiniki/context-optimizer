"""
Tests for Compressor — cluster batching, caching, API integration, fallback.

Uses a mock Anthropic client to avoid real API calls in unit tests.
Real Haiku integration is tested separately in eval.
"""

import pytest
from unittest.mock import MagicMock, patch
from anthropic import Anthropic
from dataclasses import dataclass

from src.compressor.compressor import (
    Compressor,
    CompressionCluster,
    CompressorOutput,
    CompressionResult,
    MessageWithAction,
    SelectionAction,
    _extract_text_content,
    _format_cluster_for_prompt,
    _calculate_cost,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_message(role: str, content: str) -> dict:
    """Helper to create a simple message dict."""
    return {"role": role, "content": content}


def _make_annotated(
    index: int, role: str, content: str, action: SelectionAction, score: float = 0.5
) -> MessageWithAction:
    """Helper to create an annotated message."""
    return MessageWithAction(
        index=index,
        message=_make_message(role, content),
        action=action,
        score=score,
    )


def _mock_haiku_response(summary_text: str, input_tokens: int = 150, output_tokens: int = 30):
    """Create a mock Anthropic API response."""
    mock_resp = MagicMock()
    mock_content = MagicMock()
    mock_content.text = summary_text
    mock_resp.content = [mock_content]
    mock_resp.usage = MagicMock()
    mock_resp.usage.input_tokens = input_tokens
    mock_resp.usage.output_tokens = output_tokens
    return mock_resp


@pytest.fixture
def mock_client():
    """Create a mock Anthropic client that returns controlled responses."""
    client = MagicMock()
    client.messages.create.return_value = _mock_haiku_response(
        "Team discussed database options and leaned toward PostgreSQL."
    )
    return client


@pytest.fixture
def compressor(mock_client):
    """Create a Compressor with mock client and empty cache. Disable skips for testing."""
    return Compressor(client=mock_client, cache={}, min_cluster_tokens=0, cost_aware=False)

# ---------------------------------------------------------------------------
# Cluster Building Tests
# ---------------------------------------------------------------------------

class TestClusterBuilding:
    """Tests for grouping consecutive COMPRESS messages into clusters."""

    def test_single_cluster_consecutive(self, compressor):
        """Consecutive COMPRESS messages become one cluster."""
        messages = [
            _make_annotated(0, "user", "hello", SelectionAction.KEEP),
            _make_annotated(1, "assistant", "general chat", SelectionAction.COMPRESS),
            _make_annotated(2, "user", "more chat", SelectionAction.COMPRESS),
            _make_annotated(3, "assistant", "still chatting", SelectionAction.COMPRESS),
            _make_annotated(4, "user", "important question", SelectionAction.KEEP),
        ]
        clusters = compressor._build_clusters(messages)
        assert len(clusters) == 1
        assert clusters[0].indices == [1, 2, 3]

    def test_multiple_clusters_non_consecutive(self, compressor):
        """Non-consecutive COMPRESS groups become separate clusters."""
        messages = [
            _make_annotated(0, "user", "chat 1", SelectionAction.COMPRESS),
            _make_annotated(1, "assistant", "chat 2", SelectionAction.COMPRESS),
            _make_annotated(2, "user", "important", SelectionAction.KEEP),
            _make_annotated(3, "assistant", "chat 3", SelectionAction.COMPRESS),
            _make_annotated(4, "user", "chat 4", SelectionAction.COMPRESS),
        ]
        clusters = compressor._build_clusters(messages)
        assert len(clusters) == 2
        assert clusters[0].indices == [0, 1]
        assert clusters[1].indices == [3, 4]

    def test_no_compress_messages(self, compressor):
        """No COMPRESS messages = no clusters."""
        messages = [
            _make_annotated(0, "user", "keep me", SelectionAction.KEEP),
            _make_annotated(1, "assistant", "drop me", SelectionAction.DROP),
        ]
        clusters = compressor._build_clusters(messages)
        assert len(clusters) == 0

    def test_all_compress(self, compressor):
        """All COMPRESS = one big cluster."""
        messages = [
            _make_annotated(i, "user" if i % 2 == 0 else "assistant",
                          f"msg {i}", SelectionAction.COMPRESS)
            for i in range(5)
        ]
        clusters = compressor._build_clusters(messages)
        assert len(clusters) == 1
        assert clusters[0].indices == [0, 1, 2, 3, 4]

    def test_single_compress_message(self, compressor):
        """Single COMPRESS message still forms a cluster."""
        messages = [
            _make_annotated(0, "user", "keep", SelectionAction.KEEP),
            _make_annotated(1, "assistant", "compress me", SelectionAction.COMPRESS),
            _make_annotated(2, "user", "keep", SelectionAction.KEEP),
        ]
        clusters = compressor._build_clusters(messages)
        assert len(clusters) == 1
        assert clusters[0].indices == [1]


# ---------------------------------------------------------------------------
# Cache Tests
# ---------------------------------------------------------------------------

class TestCaching:
    """Tests for in-memory compression cache."""

    def test_cache_miss_then_hit(self, compressor, mock_client):
        """First call is cache miss (API hit), second is cache hit (no API)."""
        messages = [
            _make_annotated(0, "user", "chat about weather", SelectionAction.COMPRESS),
            _make_annotated(1, "assistant", "nice day today", SelectionAction.COMPRESS),
        ]

        # First call — cache miss
        output1 = compressor.compress(messages)
        assert output1.cache_misses == 1
        assert output1.cache_hits == 0
        assert mock_client.messages.create.call_count == 1

        # Second call with same messages — cache hit
        output2 = compressor.compress(messages)
        assert output2.cache_hits == 1
        assert output2.cache_misses == 0
        assert mock_client.messages.create.call_count == 1  # no new API call

    def test_different_content_different_cache_key(self, compressor, mock_client):
        """Different message content produces different cache keys."""
        messages_a = [
            _make_annotated(0, "user", "talk about dogs", SelectionAction.COMPRESS),
        ]
        messages_b = [
            _make_annotated(0, "user", "talk about cats", SelectionAction.COMPRESS),
        ]

        compressor.compress(messages_a)
        compressor.compress(messages_b)
        assert mock_client.messages.create.call_count == 2  # both are misses

    def test_cache_size(self, compressor, mock_client):
        """Cache grows with unique clusters."""
        for i in range(3):
            messages = [
                _make_annotated(0, "user", f"unique content {i}", SelectionAction.COMPRESS),
            ]
            compressor.compress(messages)

        assert compressor.cache_size == 3

    def test_clear_cache(self, compressor, mock_client):
        """clear_cache empties the cache."""
        messages = [
            _make_annotated(0, "user", "cache me", SelectionAction.COMPRESS),
        ]
        compressor.compress(messages)
        assert compressor.cache_size == 1

        compressor.clear_cache()
        assert compressor.cache_size == 0


# ---------------------------------------------------------------------------
# Compression Output Tests
# ---------------------------------------------------------------------------

class TestCompressionOutput:
    """Tests for compress() output structure and stats."""

    def test_output_structure(self, compressor, mock_client):
        """compress() returns well-formed CompressorOutput."""
        messages = [
            _make_annotated(0, "user", "keep me", SelectionAction.KEEP),
            _make_annotated(1, "assistant", "compress a", SelectionAction.COMPRESS),
            _make_annotated(2, "user", "compress b", SelectionAction.COMPRESS),
            _make_annotated(3, "assistant", "keep me too", SelectionAction.KEEP),
        ]

        output = compressor.compress(messages)
        assert isinstance(output, CompressorOutput)
        assert output.total_clusters == 1
        assert len(output.results) == 1
        assert output.results[0].summary == "Team discussed database options and leaned toward PostgreSQL."
        assert output.results[0].from_cache is False

    def test_cost_tracking(self, compressor, mock_client):
        """Cost is tracked per cluster and totalled."""
        mock_client.messages.create.return_value = _mock_haiku_response(
            "Summary text", input_tokens=500, output_tokens=30
        )

        messages = [
            _make_annotated(0, "user", "compress me", SelectionAction.COMPRESS),
        ]

        output = compressor.compress(messages)
        assert output.total_cost_usd > 0
        result = output.results[0]
        assert result.input_tokens == 500
        assert result.output_tokens == 30
        assert result.cost_usd == _calculate_cost(500, 30)

    def test_latency_tracking(self, compressor, mock_client):
        """Latency is tracked per cluster."""
        messages = [
            _make_annotated(0, "user", "compress me", SelectionAction.COMPRESS),
        ]
        output = compressor.compress(messages)
        assert output.results[0].latency_ms >= 0
        assert output.total_latency_ms >= 0

    def test_empty_input(self, compressor):
        """No messages → empty output, no API calls."""
        output = compressor.compress([])
        assert output.total_clusters == 0
        assert len(output.results) == 0


# ---------------------------------------------------------------------------
# Summary Lookup Tests
# ---------------------------------------------------------------------------

class TestSummaryLookup:
    """Tests for get_summary_for_index and get_summary_for_indices."""

    def test_get_summary_for_index(self, compressor, mock_client):
        """Can look up which summary covers a specific message index."""
        messages = [
            _make_annotated(0, "user", "keep", SelectionAction.KEEP),
            _make_annotated(1, "assistant", "compress a", SelectionAction.COMPRESS),
            _make_annotated(2, "user", "compress b", SelectionAction.COMPRESS),
            _make_annotated(3, "assistant", "keep", SelectionAction.KEEP),
        ]

        output = compressor.compress(messages)
        assert compressor.get_summary_for_index(output, 1) is not None
        assert compressor.get_summary_for_index(output, 2) is not None
        assert compressor.get_summary_for_index(output, 0) is None  # KEEP, not in any cluster
        assert compressor.get_summary_for_index(output, 3) is None

    def test_get_summary_for_indices(self, compressor, mock_client):
        """Can look up summary by a set of indices."""
        messages = [
            _make_annotated(1, "assistant", "compress a", SelectionAction.COMPRESS),
            _make_annotated(2, "user", "compress b", SelectionAction.COMPRESS),
        ]

        output = compressor.compress(messages)
        summary = compressor.get_summary_for_indices(output, {1, 2})
        assert summary is not None

    def test_lookup_miss(self, compressor, mock_client):
        """Lookup for non-existent index returns None."""
        output = CompressorOutput()
        assert compressor.get_summary_for_index(output, 99) is None


# ---------------------------------------------------------------------------
# Fallback Tests
# ---------------------------------------------------------------------------

class TestFallback:
    """Tests for graceful degradation when Haiku API fails."""

    def test_api_failure_uses_fallback(self):
        """If Haiku call raises, compressor falls back to truncated concatenation."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API timeout")

        compressor = Compressor(client=mock_client, cache={}, min_cluster_tokens=0, cost_aware=False)
        messages = [
            _make_annotated(0, "user", "first message about planning", SelectionAction.COMPRESS),
            _make_annotated(1, "assistant", "second message about planning", SelectionAction.COMPRESS),
        ]

        output = compressor.compress(messages)
        assert output.total_clusters == 1
        result = output.results[0]
        assert result.from_cache is False
        # Fallback should contain some text from original messages
        assert "first message" in result.summary or "planning" in result.summary

    def test_fallback_cached(self):
        """Fallback result is also cached to avoid repeated failures."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API down")

        compressor = Compressor(client=mock_client, cache={}, min_cluster_tokens=0, cost_aware=False)
        messages = [
            _make_annotated(0, "user", "failing message", SelectionAction.COMPRESS),
        ]

        compressor.compress(messages)
        assert compressor.cache_size == 1  # fallback was cached

        # Second call hits cache, no new API attempt
        output2 = compressor.compress(messages)
        assert output2.cache_hits == 1


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for helper functions."""

    def test_extract_text_string(self):
        assert _extract_text_content({"role": "user", "content": "hello"}) == "hello"

    def test_extract_text_structured(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Here's the result"},
                {"type": "tool_use", "name": "search", "input": {"q": "test"}},
            ]
        }
        text = _extract_text_content(msg)
        assert "Here's the result" in text
        assert "search" in text

    def test_extract_text_empty(self):
        assert _extract_text_content({"role": "user"}) == ""
        assert _extract_text_content({"role": "user", "content": ""}) == ""

    def test_format_cluster(self):
        cluster = CompressionCluster(
            indices=[0, 1],
            messages=[
                {"role": "user", "content": "What about Redis?"},
                {"role": "assistant", "content": "Redis would work for caching."},
            ],
        )
        formatted = _format_cluster_for_prompt(cluster)
        assert "[user]: What about Redis?" in formatted
        assert "[assistant]: Redis would work for caching." in formatted

    def test_cost_calculation(self):
        cost = _calculate_cost(1000, 100)
        expected = (1000 / 1000) * 0.00025 + (100 / 1000) * 0.00125
        assert abs(cost - expected) < 1e-10

    def test_cost_zero_tokens(self):
        assert _calculate_cost(0, 0) == 0.0


# ---------------------------------------------------------------------------
# Cache Key Determinism
# ---------------------------------------------------------------------------

class TestCacheKey:
    """Tests for hash-based cache key generation."""

    def test_same_content_same_key(self):
        """Identical cluster content produces identical cache key."""
        msgs = [{"role": "user", "content": "hello world"}]
        c1 = CompressionCluster(indices=[0], messages=msgs)
        c2 = CompressionCluster(indices=[0], messages=msgs)
        assert c1.cache_key == c2.cache_key

    def test_different_content_different_key(self):
        """Different content produces different cache key."""
        c1 = CompressionCluster(
            indices=[0], messages=[{"role": "user", "content": "hello"}]
        )
        c2 = CompressionCluster(
            indices=[0], messages=[{"role": "user", "content": "goodbye"}]
        )
        assert c1.cache_key != c2.cache_key

    def test_role_matters_in_key(self):
        """Different roles produce different cache keys even with same text."""
        c1 = CompressionCluster(
            indices=[0], messages=[{"role": "user", "content": "hello"}]
        )
        c2 = CompressionCluster(
            indices=[0], messages=[{"role": "assistant", "content": "hello"}]
        )
        assert c1.cache_key != c2.cache_key
