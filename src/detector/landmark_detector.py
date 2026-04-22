import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LandmarkType(Enum):
    DECISION = "decision"
    COMMITMENT = "commitment"
    ACTION_ITEM = "action_item"
    DEADLINE = "deadline"
    TOOL_CHAIN = "tool_chain"


class ProtectionLevel(Enum):
    NONE = "none"
    PROTECTED = "protected"
    CHAIN_PROTECTED = "chain_protected"


@dataclass
class LandmarkResult:
    """Detection result for a single message."""
    message_index: int
    protection: ProtectionLevel = ProtectionLevel.NONE
    landmark_types: list[LandmarkType] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)


# --- Strict pattern definitions (Pass 1) ---

DECISION_PATTERNS = [
    r"\b(?:we|I)\s+decided\b",
    r"\blet'?s\s+go\s+with\b",
    r"\bagreed\s+(?:on|to|that)\b",
    r"\bdecision\s*(?:is|was|:)\b",
    r"\bfinal\s+(?:call|answer|decision)\b",
    r"\bwe'?(?:re|ll)\s+(?:going|gonna)\s+(?:go\s+)?with\b",
    r"\bsettled\s+on\b",
    r"\blocked\s+in\b",
    r"\bconfirmed\s+(?:that|we|the)\b",
]

COMMITMENT_PATTERNS = [
    r"\bI\s*'?(?:ll|will)\s+(?:handle|take care of|do|send|write|create|build|fix|update)\b",
    r"\bI'?m\s+going\s+to\b",
    r"\bI\s+(?:can|will)\s+take\s+(?:that|this|it)\s+on\b",
    r"\bI'?ll\s+(?:get|have)\s+(?:that|this|it)\s+(?:done|ready|finished)\b",
    r"\bI\s+(?:own|volunteer|commit)\b",
    r"\btaking\s+(?:ownership|responsibility)\b",
]

ACTION_ITEM_PATTERNS = [
    r"\baction\s+item\b",
    r"\bto\s*-?\s*do\b",
    r"\bnext\s+step\b",
    r"\bfollow\s*-?\s*up\b",
    r"\btask\s*(?::|is|—)\b",
    r"\bneeds?\s+to\s+(?:be\s+)?(?:done|completed|finished|handled)\b",
    r"\bplease\s+(?:make sure|ensure|remember)\s+to\b",
    r"\bdon'?t\s+forget\s+to\b",
]

DEADLINE_PATTERNS = [
    r"\bby\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    r"\bby\s+(?:end\s+of\s+(?:day|week|month|quarter|sprint))\b",
    r"\bdue\s+(?:date|by|on)\b",
    r"\bdeadline\b",
    r"\bby\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b",
    r"\bETA\s*(?::|is|—|-|–)?\s*(?:next|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b",
    r"\bmust\s+(?:be\s+)?(?:done|ready|shipped|delivered)\s+by\b",
]

# Compile strict patterns once at module load
_COMPILED_PATTERNS: list[tuple[re.Pattern, LandmarkType]] = []
for patterns, ltype in [
    (DECISION_PATTERNS, LandmarkType.DECISION),
    (COMMITMENT_PATTERNS, LandmarkType.COMMITMENT),
    (ACTION_ITEM_PATTERNS, LandmarkType.ACTION_ITEM),
    (DEADLINE_PATTERNS, LandmarkType.DEADLINE),
]:
    for pattern in patterns:
        _COMPILED_PATTERNS.append((re.compile(pattern, re.IGNORECASE), ltype))


# --- Fuzzy keyword proximity patterns (Pass 2) ---
# These catch informal language that strict regexes miss.
# Each is a tuple: (list of keyword sets, LandmarkType)
# A message matches if ALL keywords from any set appear within the text.

FUZZY_DECISION_KEYWORDS = [
    {"go", "ahead", "with"},
    {"lock", "that", "in"},
    {"moving", "forward", "with"},
    {"stick", "with"},
    {"chosen", "approach"},
    {"pick", "option"},
    {"going", "with", "option"},
    {"that's", "final"},
    {"approved"},
    {"greenlit"},
    {"green", "light"},
    {"sign", "off"},
    {"signed", "off"},
]

FUZZY_COMMITMENT_KEYWORDS = [
    {"i'll", "make", "sure"},
    {"i'll", "get", "it"},
    {"on", "it"},  # "I'm on it"
    {"i'll", "own"},
    {"leave", "it", "to", "me"},
    {"my", "responsibility"},
    {"i'll", "take", "care"},
    {"count", "on", "me"},
    {"i'll", "sort"},
    {"i'll", "figure"},
]

FUZZY_ACTION_KEYWORDS = [
    {"need", "to", "make", "sure"},
    {"remember", "to"},
    {"should", "also"},
    {"let's", "not", "forget"},
    {"important", "to", "note"},
    {"key", "takeaway"},
    {"make", "note"},
    {"write", "down"},
    {"track", "this"},
]

FUZZY_DEADLINE_KEYWORDS = [
    {"before", "launch"},
    {"before", "release"},
    {"ship", "date"},
    {"target", "date"},
    {"no", "later", "than"},
    {"time", "frame"},
    {"timeframe"},
    {"due", "next"},
    {"needs", "to", "be", "ready"},
    {"before", "next", "sprint"},
]

_FUZZY_PATTERNS: list[tuple[list[set[str]], LandmarkType]] = [
    (FUZZY_DECISION_KEYWORDS, LandmarkType.DECISION),
    (FUZZY_COMMITMENT_KEYWORDS, LandmarkType.COMMITMENT),
    (FUZZY_ACTION_KEYWORDS, LandmarkType.ACTION_ITEM),
    (FUZZY_DEADLINE_KEYWORDS, LandmarkType.DEADLINE),
]


def _extract_text(message: dict) -> str:
    """Extract searchable text from a message, handling both string and structured content."""
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
                    parts.append(block.get("name", ""))
                    parts.append(str(block.get("input", "")))
                elif block.get("type") == "tool_result":
                    parts.append(str(block.get("content", "")))
        return " ".join(parts)

    return str(content)


def _fuzzy_match(text: str, keyword_sets: list[set[str]]) -> Optional[str]:
    """
    Check if text contains all keywords from any keyword set.
    Returns the matched keyword set as a string, or None.
    """
    text_lower = text.lower()
    # Tokenize once for efficiency
    words = set(re.findall(r"[a-z']+", text_lower))

    for kw_set in keyword_sets:
        if kw_set.issubset(words):
            return " + ".join(sorted(kw_set))

    return None


def _has_tool_use(message: dict) -> bool:
    """Check if message contains a tool_use block."""
    content = message.get("content", "")
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_use"
            for b in content
        )
    return False


def _has_tool_result(message: dict) -> bool:
    """Check if message contains a tool_result block."""
    content = message.get("content", "")
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )
    return False


def detect_landmarks(messages: list[dict]) -> list[LandmarkResult]:
    """
    Scan all messages and return landmark detection results.

    Two-pass detection:
      Pass 1: Strict regex patterns (high confidence)
      Pass 2: Fuzzy keyword proximity (catches informal language)

    Args:
        messages: List of conversation messages, each with 'role' and 'content' keys.

    Returns:
        List of LandmarkResult, one per message, in same order as input.
    """
    results = []

    for i, message in enumerate(messages):
        result = LandmarkResult(message_index=i)
        text = _extract_text(message)

        # --- Pass 1: Strict regex patterns ---
        seen_types = set()
        for pattern, ltype in _COMPILED_PATTERNS:
            match = pattern.search(text)
            if match and ltype not in seen_types:
                seen_types.add(ltype)
                result.landmark_types.append(ltype)
                result.matched_patterns.append(match.group())

        # --- Pass 2: Fuzzy keyword proximity ---
        # Only check fuzzy if strict didn't already catch this type
        for keyword_sets, ltype in _FUZZY_PATTERNS:
            if ltype not in seen_types:
                fuzzy_match = _fuzzy_match(text, keyword_sets)
                if fuzzy_match:
                    seen_types.add(ltype)
                    result.landmark_types.append(ltype)
                    result.matched_patterns.append(f"fuzzy:{fuzzy_match}")

        if result.landmark_types:
            result.protection = ProtectionLevel.PROTECTED

        results.append(result)

    # --- Tool-chain detection ---
    for i, message in enumerate(messages):
        if _has_tool_use(message):
            if results[i].protection == ProtectionLevel.NONE:
                results[i].protection = ProtectionLevel.CHAIN_PROTECTED
                results[i].landmark_types.append(LandmarkType.TOOL_CHAIN)

            for j in range(i + 1, min(i + 3, len(messages))):
                if _has_tool_result(messages[j]):
                    if results[j].protection == ProtectionLevel.NONE:
                        results[j].protection = ProtectionLevel.CHAIN_PROTECTED
                        results[j].landmark_types.append(LandmarkType.TOOL_CHAIN)
                    break

    return results


def get_protected_indices(results: list[LandmarkResult]) -> set[int]:
    """Return indices of all messages with any protection level."""
    return {
        r.message_index for r in results
        if r.protection != ProtectionLevel.NONE
    }


def get_landmark_summary(results: list[LandmarkResult]) -> dict:
    """Return summary stats for logging/debugging."""
    protected = [r for r in results if r.protection == ProtectionLevel.PROTECTED]
    chain_protected = [r for r in results if r.protection == ProtectionLevel.CHAIN_PROTECTED]

    type_counts = {}
    for r in results:
        for lt in r.landmark_types:
            type_counts[lt.value] = type_counts.get(lt.value, 0) + 1

    return {
        "total_messages": len(results),
        "protected_count": len(protected),
        "chain_protected_count": len(chain_protected),
        "landmark_type_counts": type_counts,
    }
