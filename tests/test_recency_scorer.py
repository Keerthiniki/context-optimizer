"""Tests for Recency Scorer — exponential decay."""

import math
import pytest
from src.scorer.recency_scorer import score_recency


def test_last_message_scores_one():
    scores = score_recency(10)
    assert scores[-1] == 1.0


def test_scores_increase_toward_end():
    scores = score_recency(20)
    for i in range(len(scores) - 1):
        assert scores[i] < scores[i + 1]


def test_known_decay_values():
    """Verify exact decay at known positions with λ=0.1."""
    scores = score_recency(51, lambda_decay=0.1)

    # Last message (distance 0) = 1.0
    assert scores[50] == 1.0

    # 10 back (distance 10) ≈ e^(-1) ≈ 0.3679
    assert abs(scores[40] - math.exp(-1.0)) < 0.001

    # 20 back (distance 20) ≈ e^(-2) ≈ 0.1353
    assert abs(scores[30] - math.exp(-2.0)) < 0.001

    # 50 back (distance 50) ≈ e^(-5) ≈ 0.0067
    assert abs(scores[0] - math.exp(-5.0)) < 0.001


def test_custom_lambda():
    """Higher lambda = faster decay."""
    slow = score_recency(20, lambda_decay=0.05)
    fast = score_recency(20, lambda_decay=0.2)

    # Both have last message = 1.0
    assert slow[-1] == 1.0
    assert fast[-1] == 1.0

    # At position 0 (oldest), slow decay retains more score
    assert slow[0] > fast[0]


def test_single_message():
    scores = score_recency(1)
    assert scores == [1.0]


def test_empty():
    scores = score_recency(0)
    assert scores == []


def test_all_scores_between_zero_and_one():
    scores = score_recency(100, lambda_decay=0.1)
    for s in scores:
        assert 0.0 < s <= 1.0


def test_two_messages():
    scores = score_recency(2, lambda_decay=0.1)
    assert scores[1] == 1.0
    assert abs(scores[0] - math.exp(-0.1)) < 0.001
