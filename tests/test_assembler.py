"""
Tests for Assembler — thread reconstruction, summary injection, role alternation.
"""

import pytest

from src.assembler.assembler import Assembler, AssembledMessage, AssemblerOutput
from src.compressor.compressor import (
    CompressorOutput,
    CompressionResult,
    CompressionCluster,
    MessageWithAction,
    SelectionAction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_annotated(
    index: int, role: str, content: str, action: SelectionAction
) -> MessageWithAction:
    return MessageWithAction(
        index=index,
        message={"role": role, "content": content},
        action=action,
    )


def _make_compressor_output(clusters: list[tuple[list[int], list[dict], str]]) -> CompressorOutput:
    """
    Helper to build CompressorOutput.
    clusters: list of (indices, messages, summary_text)
    """
    results = []
    for indices, messages, summary in clusters:
        cluster = CompressionCluster(indices=indices, messages=messages)
        results.append(CompressionResult(cluster=cluster, summary=summary))
    return CompressorOutput(results=results, total_clusters=len(results))


# ---------------------------------------------------------------------------
# Basic Assembly Tests
# ---------------------------------------------------------------------------

class TestBasicAssembly:

    def test_all_keep(self):
        """All KEEP messages pass through verbatim."""
        messages = [
            _make_annotated(0, "user", "Hello", SelectionAction.KEEP),
            _make_annotated(1, "assistant", "Hi there", SelectionAction.KEEP),
            _make_annotated(2, "user", "Question?", SelectionAction.KEEP),
        ]
        assembler = Assembler()
        output = assembler.assemble(messages, CompressorOutput())

        assert len(output.raw_messages) == 3
        assert output.total_kept == 3
        assert output.total_compressed == 0
        assert output.total_dropped == 0
        assert output.raw_messages[0]["content"] == "Hello"
        assert output.raw_messages[1]["content"] == "Hi there"

    def test_all_drop(self):
        """All DROP messages produce empty output."""
        messages = [
            _make_annotated(0, "user", "noise", SelectionAction.DROP),
            _make_annotated(1, "assistant", "noise", SelectionAction.DROP),
        ]
        assembler = Assembler()
        output = assembler.assemble(messages, CompressorOutput())

        assert len(output.raw_messages) == 0
        assert output.total_dropped == 2

    def test_empty_input(self):
        """Empty input produces empty output."""
        assembler = Assembler()
        output = assembler.assemble([], CompressorOutput())
        assert len(output.raw_messages) == 0

    def test_message_order_preserved(self):
        """Messages maintain their original order."""
        messages = [
            _make_annotated(0, "user", "first", SelectionAction.KEEP),
            _make_annotated(1, "assistant", "second", SelectionAction.KEEP),
            _make_annotated(2, "user", "third", SelectionAction.KEEP),
            _make_annotated(3, "assistant", "fourth", SelectionAction.KEEP),
        ]
        assembler = Assembler()
        output = assembler.assemble(messages, CompressorOutput())

        contents = [m["content"] for m in output.raw_messages]
        assert contents == ["first", "second", "third", "fourth"]


# ---------------------------------------------------------------------------
# Summary Injection Tests
# ---------------------------------------------------------------------------

class TestSummaryInjection:

    def test_compress_cluster_gets_summary(self):
        """COMPRESS cluster is replaced by a single [SUMMARY] message."""
        messages = [
            _make_annotated(0, "user", "Hello", SelectionAction.KEEP),
            _make_annotated(1, "assistant", "chat a", SelectionAction.COMPRESS),
            _make_annotated(2, "user", "chat b", SelectionAction.COMPRESS),
            _make_annotated(3, "assistant", "Important answer", SelectionAction.KEEP),
        ]

        comp_output = _make_compressor_output([
            ([1, 2],
             [{"role": "assistant", "content": "chat a"},
              {"role": "user", "content": "chat b"}],
             "General pleasantries exchanged.")
        ])

        assembler = Assembler()
        output = assembler.assemble(messages, comp_output)

        assert output.summaries_injected == 1
        assert output.total_compressed == 2
        # Find the summary message
        summaries = [m for m in output.messages if m.is_summary]
        assert len(summaries) == 1
        assert "General pleasantries exchanged." in summaries[0].content

    def test_multiple_clusters_get_separate_summaries(self):
        """Two non-consecutive clusters each get their own summary."""
        messages = [
            _make_annotated(0, "user", "chat 1", SelectionAction.COMPRESS),
            _make_annotated(1, "assistant", "chat 2", SelectionAction.COMPRESS),
            _make_annotated(2, "user", "important", SelectionAction.KEEP),
            _make_annotated(3, "assistant", "chat 3", SelectionAction.COMPRESS),
            _make_annotated(4, "user", "chat 4", SelectionAction.COMPRESS),
        ]

        comp_output = _make_compressor_output([
            ([0, 1],
             [{"role": "user", "content": "chat 1"},
              {"role": "assistant", "content": "chat 2"}],
             "First cluster summary."),
            ([3, 4],
             [{"role": "assistant", "content": "chat 3"},
              {"role": "user", "content": "chat 4"}],
             "Second cluster summary."),
        ])

        assembler = Assembler()
        output = assembler.assemble(messages, comp_output)

        # Both summaries injected (pre-merge count)
        assert output.summaries_injected == 2
        # After role alternation fix, same-role messages merge —
        # both summaries and the KEEP are all "user" role, so they merge.
        # Verify both summary texts appear in the merged output.
        all_content = " ".join(m["content"] for m in output.raw_messages)
        assert "First cluster summary." in all_content
        assert "Second cluster summary." in all_content
        assert "important" in all_content

    def test_summary_not_duplicated_per_cluster_message(self):
        """A cluster of 3 messages emits ONE summary, not three."""
        messages = [
            _make_annotated(0, "user", "a", SelectionAction.COMPRESS),
            _make_annotated(1, "assistant", "b", SelectionAction.COMPRESS),
            _make_annotated(2, "user", "c", SelectionAction.COMPRESS),
        ]

        comp_output = _make_compressor_output([
            ([0, 1, 2],
             [{"role": "user", "content": "a"},
              {"role": "assistant", "content": "b"},
              {"role": "user", "content": "c"}],
             "Three messages summarised."),
        ])

        assembler = Assembler()
        output = assembler.assemble(messages, comp_output)

        assert output.summaries_injected == 1
        assert output.total_compressed == 3

    def test_summary_format(self):
        """Summary messages have [SUMMARY: ...] format."""
        messages = [
            _make_annotated(0, "user", "chat", SelectionAction.COMPRESS),
        ]

        comp_output = _make_compressor_output([
            ([0],
             [{"role": "user", "content": "chat"}],
             "Quick chat about weather."),
        ])

        assembler = Assembler()
        output = assembler.assemble(messages, comp_output)

        summary_msg = output.messages[0]
        assert summary_msg.content == "[SUMMARY: Quick chat about weather.]"


# ---------------------------------------------------------------------------
# Role Alternation Tests
# ---------------------------------------------------------------------------

class TestRoleAlternation:

    def test_consecutive_same_role_merged(self):
        """Two consecutive user messages get merged into one."""
        messages = [
            _make_annotated(0, "user", "first question", SelectionAction.KEEP),
            _make_annotated(1, "user", "follow up", SelectionAction.KEEP),
            _make_annotated(2, "assistant", "answer", SelectionAction.KEEP),
        ]
        assembler = Assembler()
        output = assembler.assemble(messages, CompressorOutput())

        assert len(output.raw_messages) == 2
        assert output.raw_messages[0]["role"] == "user"
        assert "first question" in output.raw_messages[0]["content"]
        assert "follow up" in output.raw_messages[0]["content"]
        assert output.merges_performed == 1

    def test_system_message_exempt(self):
        """System message at start doesn't trigger merge with following user."""
        messages = [
            _make_annotated(0, "system", "You are helpful", SelectionAction.KEEP),
            _make_annotated(1, "user", "Hello", SelectionAction.KEEP),
            _make_annotated(2, "assistant", "Hi", SelectionAction.KEEP),
        ]
        assembler = Assembler()
        output = assembler.assemble(messages, CompressorOutput())

        assert len(output.raw_messages) == 3
        assert output.raw_messages[0]["role"] == "system"
        assert output.merges_performed == 0

    def test_summary_merged_with_adjacent_same_role(self):
        """If summary (user role) follows a user KEEP, they merge."""
        messages = [
            _make_annotated(0, "user", "my question", SelectionAction.KEEP),
            _make_annotated(1, "user", "chat noise", SelectionAction.COMPRESS),
            _make_annotated(2, "assistant", "answer", SelectionAction.KEEP),
        ]

        comp_output = _make_compressor_output([
            ([1],
             [{"role": "user", "content": "chat noise"}],
             "Brief aside about weather."),
        ])

        assembler = Assembler()
        output = assembler.assemble(messages, comp_output)

        # user KEEP + user SUMMARY should merge
        assert output.raw_messages[0]["role"] == "user"
        assert "my question" in output.raw_messages[0]["content"]
        assert "SUMMARY" in output.raw_messages[0]["content"]

    def test_no_merge_needed(self):
        """Proper alternation needs no merging."""
        messages = [
            _make_annotated(0, "user", "q1", SelectionAction.KEEP),
            _make_annotated(1, "assistant", "a1", SelectionAction.KEEP),
            _make_annotated(2, "user", "q2", SelectionAction.KEEP),
            _make_annotated(3, "assistant", "a2", SelectionAction.KEEP),
        ]
        assembler = Assembler()
        output = assembler.assemble(messages, CompressorOutput())

        assert output.merges_performed == 0
        assert len(output.raw_messages) == 4


# ---------------------------------------------------------------------------
# Mixed Scenario Tests
# ---------------------------------------------------------------------------

class TestMixedScenarios:

    def test_full_pipeline_scenario(self):
        """Realistic mix of KEEP, COMPRESS, and DROP."""
        messages = [
            _make_annotated(0, "system", "You are helpful", SelectionAction.KEEP),
            _make_annotated(1, "user", "Let's plan the project", SelectionAction.KEEP),
            _make_annotated(2, "assistant", "Sure, what's the scope?", SelectionAction.KEEP),
            _make_annotated(3, "user", "chatty filler", SelectionAction.COMPRESS),
            _make_annotated(4, "assistant", "more filler", SelectionAction.COMPRESS),
            _make_annotated(5, "user", "noise", SelectionAction.DROP),
            _make_annotated(6, "assistant", "noise too", SelectionAction.DROP),
            _make_annotated(7, "user", "We decided to use PostgreSQL", SelectionAction.KEEP),
            _make_annotated(8, "assistant", "Great choice.", SelectionAction.KEEP),
        ]

        comp_output = _make_compressor_output([
            ([3, 4],
             [{"role": "user", "content": "chatty filler"},
              {"role": "assistant", "content": "more filler"}],
             "Discussed project timeline briefly."),
        ])

        assembler = Assembler()
        output = assembler.assemble(messages, comp_output)

        assert output.total_kept == 5
        assert output.total_compressed == 2
        assert output.total_dropped == 2
        assert output.summaries_injected == 1

        # Verify system message is first
        assert output.raw_messages[0]["role"] == "system"

        # Verify decision message is preserved
        decision_contents = [m["content"] for m in output.raw_messages]
        assert any("PostgreSQL" in c for c in decision_contents)

    def test_drop_between_keeps_maintains_alternation(self):
        """Dropping messages between KEEPs can create same-role adjacency — handled."""
        messages = [
            _make_annotated(0, "user", "question", SelectionAction.KEEP),
            _make_annotated(1, "assistant", "dropped answer", SelectionAction.DROP),
            _make_annotated(2, "user", "follow up", SelectionAction.KEEP),
            _make_annotated(3, "assistant", "real answer", SelectionAction.KEEP),
        ]
        assembler = Assembler()
        output = assembler.assemble(messages, CompressorOutput())

        # Two user messages became adjacent after drop — should be merged
        assert output.raw_messages[0]["role"] == "user"
        assert output.raw_messages[1]["role"] == "assistant"
        assert output.merges_performed == 1


# ---------------------------------------------------------------------------
# Output Structure Tests
# ---------------------------------------------------------------------------

class TestOutputStructure:

    def test_raw_messages_match_assembled(self):
        """raw_messages and messages lists are consistent."""
        messages = [
            _make_annotated(0, "user", "hello", SelectionAction.KEEP),
            _make_annotated(1, "assistant", "hi", SelectionAction.KEEP),
        ]
        assembler = Assembler()
        output = assembler.assemble(messages, CompressorOutput())

        assert len(output.raw_messages) == len(output.messages)
        for raw, assembled in zip(output.raw_messages, output.messages):
            assert raw["role"] == assembled.role
            assert raw["content"] == assembled.content

    def test_original_index_tracked(self):
        """KEEP messages track their original conversation index."""
        messages = [
            _make_annotated(5, "user", "from position 5", SelectionAction.KEEP),
            _make_annotated(10, "assistant", "from position 10", SelectionAction.KEEP),
        ]
        assembler = Assembler()
        output = assembler.assemble(messages, CompressorOutput())

        assert output.messages[0].original_index == 5
        assert output.messages[1].original_index == 10

    def test_summary_covers_tracks_indices(self):
        """Summary messages track which original indices they cover."""
        messages = [
            _make_annotated(3, "user", "a", SelectionAction.COMPRESS),
            _make_annotated(4, "assistant", "b", SelectionAction.COMPRESS),
            _make_annotated(5, "user", "c", SelectionAction.COMPRESS),
        ]

        comp_output = _make_compressor_output([
            ([3, 4, 5],
             [{"role": "user", "content": "a"},
              {"role": "assistant", "content": "b"},
              {"role": "user", "content": "c"}],
             "Three-way discussion."),
        ])

        assembler = Assembler()
        output = assembler.assemble(messages, comp_output)

        summaries = [m for m in output.messages if m.is_summary]
        assert len(summaries) == 1
        assert set(summaries[0].summary_covers) == {3, 4, 5}
