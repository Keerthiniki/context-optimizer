"""Tests for Landmark Detector — covers all landmark types and tool chains."""

import pytest
from src.detector.landmark_detector import (
    detect_landmarks,
    get_protected_indices,
    get_landmark_summary,
    LandmarkType,
    ProtectionLevel,
)


# --- Decision detection ---

@pytest.mark.parametrize("text", [
    "We decided to use PostgreSQL for the main database.",
    "Let's go with React for the frontend.",
    "Agreed on a two-week sprint cycle.",
    "The final decision is to ship on Monday.",
    "We're going with option B.",
    "Settled on the microservices approach.",
    "Locked in the pricing at $49/month.",
    "Confirmed that we'll use AWS.",
])
def test_detects_decisions(text):
    messages = [{"role": "user", "content": text}]
    results = detect_landmarks(messages)
    assert results[0].protection == ProtectionLevel.PROTECTED
    assert LandmarkType.DECISION in results[0].landmark_types


# --- Commitment detection ---

@pytest.mark.parametrize("text", [
    "I'll handle the deployment script.",
    "I will send the report by Friday.",
    "I'm going to refactor the auth module.",
    "I can take that on.",
    "I'll get that done before standup.",
    "Taking ownership of the migration.",
])
def test_detects_commitments(text):
    messages = [{"role": "user", "content": text}]
    results = detect_landmarks(messages)
    assert results[0].protection == ProtectionLevel.PROTECTED
    assert LandmarkType.COMMITMENT in results[0].landmark_types


# --- Action item detection ---

@pytest.mark.parametrize("text", [
    "Action item: update the CI pipeline.",
    "TODO: write integration tests.",
    "Next step is to deploy to staging.",
    "Follow-up with the design team on mockups.",
    "This needs to be done before release.",
    "Please make sure to update the docs.",
    "Don't forget to notify the stakeholders.",
])
def test_detects_action_items(text):
    messages = [{"role": "user", "content": text}]
    results = detect_landmarks(messages)
    assert results[0].protection == ProtectionLevel.PROTECTED
    assert LandmarkType.ACTION_ITEM in results[0].landmark_types


# --- Deadline detection ---

@pytest.mark.parametrize("text", [
    "This needs to be ready by Friday.",
    "Due date is March 15th.",
    "The deadline for submissions is next week.",
    "Must be shipped by end of sprint.",
    "ETA: next Tuesday.",
    "By end of day we need the final version.",
])
def test_detects_deadlines(text):
    messages = [{"role": "user", "content": text}]
    results = detect_landmarks(messages)
    assert results[0].protection == ProtectionLevel.PROTECTED
    assert LandmarkType.DEADLINE in results[0].landmark_types


# --- Non-landmark messages stay unprotected ---

@pytest.mark.parametrize("text", [
    "Sounds good.",
    "Thanks for the update.",
    "Can you explain that again?",
    "Interesting point.",
    "Let me think about it.",
    "ok",
])
def test_non_landmarks_unprotected(text):
    messages = [{"role": "user", "content": text}]
    results = detect_landmarks(messages)
    assert results[0].protection == ProtectionLevel.NONE
    assert results[0].landmark_types == []


# --- Tool-chain detection ---

def test_tool_chain_detection():
    messages = [
        {"role": "user", "content": "Search for the latest sales data."},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "search", "input": {"query": "sales"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "Q3 sales: $1.2M"}
            ],
        },
        {"role": "assistant", "content": "Based on the search, Q3 sales were $1.2M."},
    ]
    results = detect_landmarks(messages)

    # tool_use message → CHAIN_PROTECTED
    assert results[1].protection == ProtectionLevel.CHAIN_PROTECTED
    assert LandmarkType.TOOL_CHAIN in results[1].landmark_types

    # tool_result message → CHAIN_PROTECTED
    assert results[2].protection == ProtectionLevel.CHAIN_PROTECTED
    assert LandmarkType.TOOL_CHAIN in results[2].landmark_types

    # Other messages unaffected
    assert results[0].protection == ProtectionLevel.NONE
    assert results[3].protection == ProtectionLevel.NONE


# --- Mixed: landmark inside a tool chain gets highest protection ---

def test_landmark_in_tool_chain_keeps_protected():
    """If a message is both a landmark and part of a tool chain,
    PROTECTED takes priority since it was set first by pattern matching."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "We decided to use PostgreSQL."},
                {"type": "tool_use", "id": "call_1", "name": "db_setup", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "DB created."}
            ],
        },
    ]
    results = detect_landmarks(messages)

    # First message: pattern match sets PROTECTED, tool chain detection
    # sees protection is already set so doesn't downgrade to CHAIN_PROTECTED
    assert results[0].protection == ProtectionLevel.PROTECTED
    assert LandmarkType.DECISION in results[0].landmark_types


# --- Structured content handling ---

def test_structured_text_content():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "We decided to use Kubernetes."},
            ],
        },
    ]
    results = detect_landmarks(messages)
    assert results[0].protection == ProtectionLevel.PROTECTED
    assert LandmarkType.DECISION in results[0].landmark_types


# --- Helper functions ---

def test_get_protected_indices():
    messages = [
        {"role": "user", "content": "Let's go with option A."},
        {"role": "assistant", "content": "Sounds good."},
        {"role": "user", "content": "I'll handle the deployment."},
    ]
    results = detect_landmarks(messages)
    indices = get_protected_indices(results)
    assert 0 in indices  # decision
    assert 2 in indices  # commitment
    assert 1 not in indices  # filler


def test_get_landmark_summary():
    messages = [
        {"role": "user", "content": "We decided on PostgreSQL."},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Action item: write the migration."},
    ]
    results = detect_landmarks(messages)
    summary = get_landmark_summary(results)

    assert summary["total_messages"] == 3
    assert summary["protected_count"] == 2
    assert summary["landmark_type_counts"]["decision"] >= 1
    assert summary["landmark_type_counts"]["action_item"] >= 1


# --- Empty and edge cases ---

def test_empty_messages():
    results = detect_landmarks([])
    assert results == []


def test_empty_content():
    messages = [{"role": "user", "content": ""}]
    results = detect_landmarks(messages)
    assert results[0].protection == ProtectionLevel.NONE


def test_multiple_landmark_types_single_message():
    """A message can contain multiple landmark types."""
    text = "We decided to use Redis. Action item: I'll handle it by Friday."
    messages = [{"role": "user", "content": text}]
    results = detect_landmarks(messages)

    assert results[0].protection == ProtectionLevel.PROTECTED
    types = results[0].landmark_types
    assert LandmarkType.DECISION in types
    assert LandmarkType.ACTION_ITEM in types
    assert LandmarkType.COMMITMENT in types
    assert LandmarkType.DEADLINE in types
