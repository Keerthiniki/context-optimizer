import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import tiktoken
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class SelectionAction(Enum):
    """Mirror of selector's action enum — keep in sync."""
    KEEP = "keep"
    COMPRESS = "compress"
    DROP = "drop"


@dataclass
class MessageWithAction:
    """A message annotated with its selection action by the Selector."""
    index: int
    message: dict          # original {"role": ..., "content": ...}
    action: SelectionAction
    score: float = 0.0


@dataclass
class CompressionCluster:
    """A group of consecutive COMPRESS messages to be summarised together."""
    indices: list[int]
    messages: list[dict]
    cache_key: str = ""

    def __post_init__(self):
        self.cache_key = self._compute_hash()

    def _compute_hash(self) -> str:
        """Content-addressable hash for cache lookup."""
        content_str = "|".join(
            f"{m.get('role', '')}:{_extract_text_content(m)}"
            for m in self.messages
        )
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]


@dataclass
class CompressionResult:
    """Result of compressing one cluster."""
    cluster: CompressionCluster
    summary: str
    from_cache: bool = False
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    skipped: bool = False  # True if compression was skipped (cost-aware)


@dataclass
class CompressorOutput:
    """Full output from the compression step."""
    results: list[CompressionResult] = field(default_factory=list)
    total_clusters: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    clusters_skipped: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text_content(message: dict) -> str:
    """Extract plain text from a message (handles string and structured content)."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    parts.append(f"[tool_use: {block.get('name', '')}]")
                elif block.get("type") == "tool_result":
                    parts.append(f"[tool_result: {str(block.get('content', ''))[:100]}]")
        return " ".join(parts)
    return str(content)


def _format_cluster_for_prompt(cluster: CompressionCluster) -> str:
    """Format a cluster of messages into a readable block for the Haiku prompt."""
    lines = []
    for msg in cluster.messages:
        role = msg.get("role", "unknown")
        text = _extract_text_content(msg)
        lines.append(f"[{role}]: {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Content-type detection for selective summarisation
# ---------------------------------------------------------------------------

def _classify_cluster_content(cluster: CompressionCluster) -> str:
    """
    Classify a cluster's content to determine compression intensity.

    Returns one of:
    - "small_talk": greetings, acknowledgements, filler -> compress aggressively
    - "reasoning": analysis, comparisons, explanations -> compress lightly
    - "technical": code, configs, numbers, steps -> should not be compressed
    """
    full_text = " ".join(
        _extract_text_content(m) for m in cluster.messages
    ).lower()

    # Technical indicators
    tech_patterns = [
        r"```",
        r"https?://",
        r"[A-Z_]{3,}=",
        r"\d+\.\d+\.\d+",       # version numbers like 3.11.4
        r"\bport\b",
        r"\bconfig\b",
        r"\$\d+",                # monetary amounts
        r"\b\d+\s*(?:gb|mb|kb|ms|cpu|ram)\b",
    ]
    tech_hits = sum(1 for p in tech_patterns if re.search(p, full_text, re.IGNORECASE))
    if tech_hits >= 2:
        return "technical"

    # Reasoning indicators
    reasoning_patterns = [
        r"\bbecause\b",
        r"\btherefore\b",
        r"\bhowever\b",
        r"\bcompare\b",
        r"\btrade.?off\b",
        r"\bpros\b",
        r"\bcons\b",
        r"\bon the other hand\b",
        r"\bthe reason\b",
        r"\badvantage\b",
        r"\bdisadvantage\b",
        r"\binstead\b",
        r"\balternative\b",
        r"\bwhereas\b",
        r"\bshould we\b",
        r"\bwhat if\b",
        r"\bi think\b",
    ]
    reasoning_hits = sum(1 for p in reasoning_patterns if re.search(p, full_text))
    if reasoning_hits >= 2:
        return "reasoning"

    # Small talk indicators
    small_talk_patterns = [
        r"\b(?:ok|okay|sure|thanks|thank you|sounds good|got it|great|nice)\b",
        r"\b(?:yeah|yep|yup|right|exactly|agreed|cool|perfect)\b",
        r"\b(?:hello|hi|hey|good morning|good afternoon)\b",
        r"\b(?:no worries|no problem|np|lol|haha)\b",
    ]
    small_talk_hits = sum(1 for p in small_talk_patterns if re.search(p, full_text))
    avg_msg_len = len(full_text) / max(len(cluster.messages), 1)

    # Short messages with small talk words -> small talk
    if small_talk_hits >= 1 and avg_msg_len < 100:
        return "small_talk"

    # Default: treat as reasoning (compress lightly to be safe)
    return "reasoning"


# ---------------------------------------------------------------------------
# Haiku cost calculation
# ---------------------------------------------------------------------------

HAIKU_INPUT_COST_PER_1K = 0.00025
HAIKU_OUTPUT_COST_PER_1K = 0.00125

# Downstream Sonnet cost per 1K tokens (used for cost-aware skip)
SONNET_INPUT_COST_PER_1K = 0.003


def _calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate USD cost for a Haiku API call."""
    return (
        (input_tokens / 1000) * HAIKU_INPUT_COST_PER_1K
        + (output_tokens / 1000) * HAIKU_OUTPUT_COST_PER_1K
    )


def _estimate_compression_savings(cluster_tokens: int, summary_tokens_est: int = 30) -> float:
    """Estimate downstream Sonnet cost saved by compressing this cluster."""
    tokens_saved = max(0, cluster_tokens - summary_tokens_est)
    return (tokens_saved / 1000) * SONNET_INPUT_COST_PER_1K


# ---------------------------------------------------------------------------
# Selective compression prompts
# ---------------------------------------------------------------------------

SMALL_TALK_PROMPT = (
    "You are a conversation summariser. Compress this block into "
    "a brief 1-sentence summary. This is filler/small talk — be very concise.\n\n"
    "Rules:\n"
    "- Output ONLY the summary text, nothing else\n"
    "- Keep it under 15 words\n"
    '- Example: "Team exchanged greetings and acknowledged the update."\n'
    '- If no meaningful content, output "General discussion continued."'
)

REASONING_PROMPT = (
    "You are a conversation summariser. Compress this block into "
    "a concise summary that preserves the REASONING and LOGIC — not just the conclusion.\n\n"
    "Rules:\n"
    "- Output ONLY the summary text, nothing else\n"
    "- Keep it under 60 words\n"
    "- Preserve: the arguments made, comparisons drawn, trade-offs discussed\n"
    "- Preserve: who said what and why they said it\n"
    "- Keep specific names, numbers, and technical terms\n"
    "- If someone gave a reason for a position, include that reason"
)

TECHNICAL_PROMPT = (
    "You are a conversation summariser. This block contains technical "
    "details. Preserve specifics as much as possible.\n\n"
    "Rules:\n"
    "- Output ONLY the summary text, nothing else\n"
    "- Keep it under 80 words\n"
    "- Preserve ALL numbers, versions, config values, URLs, and commands\n"
    "- Preserve step ordering if present\n"
    "- Do not generalise specific values into vague descriptions"
)

# Map content type to prompt and max output tokens
COMPRESSION_PROMPTS = {
    "small_talk": SMALL_TALK_PROMPT,
    "reasoning": REASONING_PROMPT,
    "technical": TECHNICAL_PROMPT,
}

MAX_OUTPUT_TOKENS = {
    "small_talk": 40,
    "reasoning": 100,
    "technical": 150,
}

# Default prompt (backward compatibility)
COMPRESSION_SYSTEM_PROMPT = REASONING_PROMPT


# ---------------------------------------------------------------------------
# Compressor class
# ---------------------------------------------------------------------------

class Compressor:
    """
    Compresses low-value message clusters using Claude Haiku.

    Features:
    - Cluster batching: consecutive COMPRESS messages = 1 API call
    - Content-hash caching: same content never compressed twice
    - Cost-aware skip: if compression cost > downstream savings, skip it
    - Selective summarisation: small talk compressed aggressively, reasoning lightly
    - Small cluster skip: clusters under min_cluster_tokens kept as-is
    - Graceful fallback: API failures produce truncated summaries
    """

    def __init__(
        self,
        model: Optional[str] = None,
        client: Optional[Anthropic] = None,
        cache: Optional[dict] = None,
        min_cluster_tokens: int = 50,
        cost_aware: bool = True,
    ):
        self.model = model or os.getenv("COMPRESSION_MODEL", "claude-haiku-4-5-20251001")
        self.client = client or Anthropic()
        self._cache: dict[str, str] = cache if cache is not None else {}
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self.min_cluster_tokens = min_cluster_tokens
        self.cost_aware = cost_aware

    # --- Public API ---

    def compress(self, annotated_messages: list[MessageWithAction]) -> CompressorOutput:
        """
        Compress all COMPRESS-tagged message clusters.

        Args:
            annotated_messages: Messages with SelectionAction from the Selector,
                                ordered by original conversation position.

        Returns:
            CompressorOutput with compression results, cost, and cache stats.
        """
        clusters = self._build_clusters(annotated_messages)
        output = CompressorOutput(total_clusters=len(clusters))

        for cluster in clusters:
            result = self._compress_cluster(cluster)
            output.results.append(result)

            if result.skipped:
                output.clusters_skipped += 1
            elif result.from_cache:
                output.cache_hits += 1
            else:
                output.cache_misses += 1

            output.total_cost_usd += result.cost_usd
            output.total_latency_ms += result.latency_ms

        return output

    def get_summary_for_indices(
        self, output: CompressorOutput, indices: set[int]
    ) -> Optional[str]:
        """Look up the summary that covers a given set of message indices."""
        for result in output.results:
            if set(result.cluster.indices) & indices:
                return result.summary
        return None

    def get_summary_for_index(
        self, output: CompressorOutput, index: int
    ) -> Optional[str]:
        """Look up the summary covering a specific message index."""
        for result in output.results:
            if index in result.cluster.indices:
                return result.summary
        return None

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()

    # --- Cluster building ---

    def _build_clusters(
        self, annotated_messages: list[MessageWithAction]
    ) -> list[CompressionCluster]:
        """Group consecutive COMPRESS-action messages into clusters."""
        clusters: list[CompressionCluster] = []
        current_indices: list[int] = []
        current_messages: list[dict] = []

        for am in annotated_messages:
            if am.action == SelectionAction.COMPRESS:
                current_indices.append(am.index)
                current_messages.append(am.message)
            else:
                if current_indices:
                    clusters.append(CompressionCluster(
                        indices=list(current_indices),
                        messages=list(current_messages),
                    ))
                    current_indices = []
                    current_messages = []

        if current_indices:
            clusters.append(CompressionCluster(
                indices=list(current_indices),
                messages=list(current_messages),
            ))

        return clusters

    # --- Token counting for clusters ---

    def _count_cluster_tokens(self, cluster: CompressionCluster) -> int:
        """Count total tokens in a cluster's messages."""
        total = 0
        for msg in cluster.messages:
            text = _extract_text_content(msg)
            total += len(self._tokenizer.encode(text))
        return total

    # --- Single cluster compression ---

    def _compress_cluster(self, cluster: CompressionCluster) -> CompressionResult:
        """
        Compress a single cluster — with cost-aware and size-aware skips.

        Order of checks:
        1. Small cluster skip (< min_cluster_tokens)
        2. Cache hit
        3. Cost-aware skip (compression cost > savings)
        4. API call with content-type-specific prompt and fallback
        """
        cluster_tokens = self._count_cluster_tokens(cluster)

        # Skip small clusters — not worth an API call
        if cluster_tokens < self.min_cluster_tokens:
            fallback = self._fallback_summary(cluster)
            return CompressionResult(
                cluster=cluster,
                summary=fallback,
                skipped=True,
            )

        # Cache hit
        if cluster.cache_key in self._cache:
            return CompressionResult(
                cluster=cluster,
                summary=self._cache[cluster.cache_key],
                from_cache=True,
            )

        # Cost-aware skip — estimate if compression is worth it
        if self.cost_aware:
            est_savings = _estimate_compression_savings(cluster_tokens)
            # Estimate Haiku cost: cluster tokens as input + ~30 tokens output
            est_cost = _calculate_cost(cluster_tokens + 100, 30)  # +100 for system prompt
            if est_cost > est_savings:
                fallback = self._fallback_summary(cluster)
                self._cache[cluster.cache_key] = fallback
                return CompressionResult(
                    cluster=cluster,
                    summary=fallback,
                    skipped=True,
                )

        # Cache miss — call Haiku with content-type-specific prompt
        prompt_text = _format_cluster_for_prompt(cluster)
        content_type = _classify_cluster_content(cluster)
        system_prompt = COMPRESSION_PROMPTS.get(content_type, COMPRESSION_SYSTEM_PROMPT)
        max_output = MAX_OUTPUT_TOKENS.get(content_type, 100)

        start = time.perf_counter()

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_output,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt_text}],
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            summary = response.content[0].text.strip()
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = _calculate_cost(input_tokens, output_tokens)

            self._cache[cluster.cache_key] = summary

            return CompressionResult(
                cluster=cluster,
                summary=summary,
                from_cache=False,
                latency_ms=elapsed_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            fallback = self._fallback_summary(cluster)
            self._cache[cluster.cache_key] = fallback

            return CompressionResult(
                cluster=cluster,
                summary=fallback,
                from_cache=False,
                latency_ms=elapsed_ms,
            )

    def _fallback_summary(self, cluster: CompressionCluster) -> str:
        """
        Emergency fallback when Haiku API fails or compression is skipped.
        Concatenates first 100 chars of each message, capped at 200 chars total.
        """
        parts = []
        total_len = 0
        for msg in cluster.messages:
            text = _extract_text_content(msg)[:100]
            if total_len + len(text) > 200:
                break
            parts.append(text)
            total_len += len(text)
        return " | ".join(parts) if parts else "Discussion continued."
