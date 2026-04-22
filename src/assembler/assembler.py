from dataclasses import dataclass, field
from typing import Optional

from src.compressor.compressor import (
    CompressorOutput,
    MessageWithAction,
    SelectionAction,
    _extract_text_content,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AssembledMessage:
    """A message in the final assembled thread."""
    role: str
    content: str | list
    original_index: Optional[int] = None     # None for injected summaries
    is_summary: bool = False
    summary_covers: list[int] = field(default_factory=list)  # indices summarised


@dataclass
class AssemblerOutput:
    """Full output from the assembly step."""
    messages: list[AssembledMessage] = field(default_factory=list)
    raw_messages: list[dict] = field(default_factory=list)  # for direct API use
    total_kept: int = 0
    total_compressed: int = 0
    total_dropped: int = 0
    summaries_injected: int = 0
    merges_performed: int = 0


# ---------------------------------------------------------------------------
# Assembler class
# ---------------------------------------------------------------------------

class Assembler:
    """
    Rebuilds a valid conversation thread from Selector + Compressor output.

    Usage:
        assembler = Assembler()
        output = assembler.assemble(annotated_messages, compressor_output)
        # output.raw_messages is ready for Claude API consumption
    """

    def __init__(self, summary_role: str = "user"):
        """
        Args:
            summary_role: Role to assign to injected summary messages.
                          'user' keeps the thread valid for Claude API.
        """
        self.summary_role = summary_role

    def assemble(
        self,
        annotated_messages: list[MessageWithAction],
        compressor_output: CompressorOutput,
    ) -> AssemblerOutput:
        """
        Assemble the final conversation thread.

        Args:
            annotated_messages: Messages with SelectionAction from the Selector,
                                ordered by original conversation position.
            compressor_output: Output from the Compressor with cluster summaries.

        Returns:
            AssemblerOutput with assembled messages ready for LLM consumption.
        """
        output = AssemblerOutput()

        # --- Build index → summary mapping ---
        index_to_summary: dict[int, str] = {}
        index_to_cluster_indices: dict[int, list[int]] = {}
        for result in compressor_output.results:
            for idx in result.cluster.indices:
                index_to_summary[idx] = result.summary
                index_to_cluster_indices[idx] = result.cluster.indices

        # --- Track which clusters we've already emitted a summary for ---
        emitted_clusters: set[tuple[int, ...]] = set()

        # --- Walk through annotated messages in order ---
        assembled: list[AssembledMessage] = []

        for am in annotated_messages:
            if am.action == SelectionAction.KEEP:
                assembled.append(AssembledMessage(
                    role=am.message.get("role", "user"),
                    content=am.message.get("content", ""),
                    original_index=am.index,
                ))
                output.total_kept += 1

            elif am.action == SelectionAction.COMPRESS:
                cluster_key = tuple(index_to_cluster_indices.get(am.index, [am.index]))

                if cluster_key not in emitted_clusters:
                    summary_text = index_to_summary.get(am.index, "Discussion continued.")
                    assembled.append(AssembledMessage(
                        role=self.summary_role,
                        content=f"[SUMMARY: {summary_text}]",
                        is_summary=True,
                        summary_covers=list(cluster_key),
                    ))
                    emitted_clusters.add(cluster_key)
                    output.summaries_injected += 1

                output.total_compressed += 1

            elif am.action == SelectionAction.DROP:
                output.total_dropped += 1

        # --- Fix role alternation ---
        assembled = self._fix_role_alternation(assembled, output)

        # --- Ensure thread starts with system or user ---
        assembled = self._fix_thread_start(assembled, annotated_messages)

        # --- Build raw messages for API consumption ---
        output.messages = assembled
        output.raw_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in assembled
        ]

        return output

    def _fix_thread_start(
        self,
        messages: list[AssembledMessage],
        annotated_messages: list[MessageWithAction],
    ) -> list[AssembledMessage]:
        """
        Ensure thread starts with system or user, never assistant.

        If first message is assistant, find the earliest user message from the
        original conversation and inject it at the start. If none found,
        inject a context marker as user message.
        """
        if not messages:
            return messages

        first_role = messages[0].role
        if first_role in ("system", "user"):
            return messages

        # First message is assistant — need to prepend a user message
        # Try to find the first user message from original conversation
        first_user = None
        for am in annotated_messages:
            if am.message.get("role") == "user":
                content = am.message.get("content", "")
                if isinstance(content, str) and content.strip():
                    first_user = AssembledMessage(
                        role="user",
                        content=content,
                        original_index=am.index,
                    )
                    break

        if first_user:
            messages.insert(0, first_user)
        else:
            # Fallback — inject a context marker
            messages.insert(0, AssembledMessage(
                role="user",
                content="[Previous conversation context follows]",
                is_summary=True,
            ))

        return messages

    def _fix_role_alternation(
        self,
        messages: list[AssembledMessage],
        output: AssemblerOutput,
    ) -> list[AssembledMessage]:
        """
        Ensure no consecutive same-role messages by merging adjacent duplicates.

        Claude API requires strict user/assistant alternation. When we drop or
        compress messages, we can end up with consecutive user or consecutive
        assistant messages. This merges them into one.

        System messages at index 0 are exempt — they're always valid.
        """
        if len(messages) <= 1:
            return messages

        fixed: list[AssembledMessage] = [messages[0]]

        for msg in messages[1:]:
            prev = fixed[-1]

            # System messages don't participate in alternation
            if msg.role == "system" or prev.role == "system":
                fixed.append(msg)
                continue

            # Same role as previous — merge content
            if msg.role == prev.role:
                prev.content = self._merge_content(prev.content, msg.content)
                if msg.is_summary:
                    prev.summary_covers.extend(msg.summary_covers)
                if msg.is_summary or prev.is_summary:
                    prev.is_summary = True
                output.merges_performed += 1
            else:
                fixed.append(msg)

        return fixed

    def _merge_content(self, existing: str | list, new: str | list) -> str:
        """Merge two message contents into one."""
        existing_str = self._content_to_string(existing)
        new_str = self._content_to_string(new)
        return f"{existing_str}\n\n{new_str}"

    def _content_to_string(self, content: str | list) -> str:
        """Convert message content to string form."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return _extract_text_content({"content": content})
        return str(content)
