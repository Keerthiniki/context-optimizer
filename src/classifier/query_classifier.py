import re
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    """Output of query classification."""
    query_type: str  # "factual", "analytical", "procedural"
    confidence: str  # "high" (keyword match) or "low" (fallback)
    matched_pattern: str | None = None  # The pattern that triggered classification


# --- Pattern definitions ---

FACTUAL_PATTERNS = [
    r"\bwhat\s+did\s+we\s+(?:decide|agree|choose|pick|select)\b",
    r"\bwho\s+(?:is|was|will)\s+(?:responsible|handling|owning|leading)\b",
    r"\bwho\s+(?:owns|handles|leads|manages)\b",
    r"\bwhen\s+(?:is|was|did|will|do)\b",
    r"\bwhere\s+(?:is|are|was|were|do|did)\b",
    r"\bwhich\s+(?:one|option|approach|tool|framework|database)\b",
    r"\bdid\s+we\s+(?:decide|agree|confirm|settle)\b",
    r"\bwhat'?s\s+the\s+(?:deadline|timeline|eta|budget|status|plan)\b",
    r"\bhow\s+much\s+(?:does|did|will|is)\b",
    r"\bhow\s+many\b",
    r"\bwhat\s+(?:is|was)\s+(?:the|our)\b",
]

ANALYTICAL_PATTERNS = [
    r"\bsummar(?:ise|ize|y)\b",
    r"\bcompare\b",
    r"\bcontrast\b",
    r"\bpros\s+and\s+cons\b",
    r"\badvantages?\s+and\s+disadvantages?\b",
    r"\bwhat\s+are\s+the\s+(?:key|main|important)\s+(?:points|takeaways|themes|issues|findings|differences)\b",
    r"\boverall\b",
    r"\banalys(?:e|is|ze)\b",
    r"\breview\b",
    r"\bassess\b",
    r"\bevaluat(?:e|ion)\b",
    r"\bwhat\s+(?:do|did)\s+we\s+(?:discuss|talk|cover)\b",
    r"\bgive\s+me\s+(?:a|an)\s+(?:overview|breakdown|recap)\b",
    r"\bwhat\s+(?:are|were)\s+(?:the\s+|all\s+(?:the\s+)?)?(?:options|alternatives|points)\b",
    r"\btrade\s*-?\s*offs?\b",
]

PROCEDURAL_PATTERNS = [
    r"\bhow\s+(?:do|did|can|could|should|would)\s+(?:we|I|you)\b",
    r"\bwhat\s+are\s+the\s+steps\b",
    r"\bwalk\s+me\s+through\b",
    r"\bstep\s*-?\s*by\s*-?\s*step\b",
    r"\bhow\s+to\b",
    r"\bexplain\s+(?:how|the\s+process)\b",
    r"\bwhat'?s\s+the\s+process\b",
    r"\bguide\s+me\b",
    r"\binstructions?\s+(?:for|on|to)\b",
    r"\bprocedure\s+(?:for|to)\b",
    r"\bworkflow\s+(?:for|to)\b",
]

# Compile all patterns once at module load
_COMPILED: list[tuple[re.Pattern, str]] = []
for patterns, qtype in [
    (FACTUAL_PATTERNS, "factual"),
    (ANALYTICAL_PATTERNS, "analytical"),
    (PROCEDURAL_PATTERNS, "procedural"),
]:
    for pattern in patterns:
        _COMPILED.append((re.compile(pattern, re.IGNORECASE), qtype))


def classify_query(query: str) -> ClassificationResult:
    """
    Classify a query into factual, analytical, or procedural.

    Args:
        query: The user's current query string.

    Returns: 
        ClassificationResult with query_type, confidence, and matched pattern.
    """
    if not query.strip():
        return ClassificationResult(
            query_type="analytical",
            confidence="low",
            matched_pattern=None,
        )

    # Count matches per type — take the type with most matches.
    # Analytical and procedural are checked first because their patterns
    # are more specific. Factual patterns like "what is/are the" are broad
    # and would otherwise swallow analytical/procedural queries.
    type_matches: dict[str, list[str]] = {
        "factual": [],
        "analytical": [],
        "procedural": [],
    }

    for pattern, qtype in _COMPILED:
        match = pattern.search(query)
        if match:
            type_matches[qtype].append(match.group())

    # Priority: if analytical or procedural matched, prefer them over factual
    # because their patterns are more specific.
    for priority_type in ["procedural", "analytical", "factual"]:
        if type_matches[priority_type]:
            return ClassificationResult(
                query_type=priority_type,
                confidence="high",
                matched_pattern=type_matches[priority_type][0],
            )

    # No pattern matched — default to analytical (broadest coverage)
    return ClassificationResult(
        query_type="analytical",
        confidence="low",
        matched_pattern=None,
    )
