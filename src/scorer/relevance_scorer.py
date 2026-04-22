from dataclasses import dataclass
from src.scorer.bm25_scorer import BM25Scorer
from src.scorer.semantic_scorer import SemanticScorer
from src.scorer.recency_scorer import score_recency
from src.detector.landmark_detector import (
    detect_landmarks,
    LandmarkResult,
    ProtectionLevel,
)


@dataclass
class ScoringWeights:
    """Weight configuration for the four scoring signals."""
    keyword: float = 0.25
    semantic: float = 0.25
    recency: float = 0.25
    landmark: float = 0.25

    def __post_init__(self):
        total = self.keyword + self.semantic + self.recency + self.landmark
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


# Preset weight configurations per query type
FACTUAL_WEIGHTS = ScoringWeights(keyword=0.3, semantic=0.2, recency=0.2, landmark=0.3)
ANALYTICAL_WEIGHTS = ScoringWeights(keyword=0.2, semantic=0.3, recency=0.2, landmark=0.3)
PROCEDURAL_WEIGHTS = ScoringWeights(keyword=0.2, semantic=0.3, recency=0.3, landmark=0.2)


LANDMARK_BOOST = 2.0  # Multiplier applied to landmark messages


@dataclass
class MessageScore:
    """Full scoring breakdown for a single message."""
    index: int
    keyword_score: float
    semantic_score: float
    recency_score: float
    landmark_boost: float
    final_score: float
    protection: ProtectionLevel


def score_messages(
    messages: list[dict],
    query: str,
    weights: ScoringWeights | None = None,
    lambda_decay: float = 0.1,
) -> list[MessageScore]:
    """
    Score all messages against the query using all four signals.

    Args:
        messages: List of conversation messages with 'role' and 'content'.
        query: Current user query.
        weights: Scoring weights. Defaults to equal weights.
        lambda_decay: Recency decay rate.

    Returns:
        List of MessageScore, one per message, in same order as input.
    """
    if not messages:
        return []

    if weights is None:
        weights = ScoringWeights()

    # 1. BM25 keyword scores
    bm25 = BM25Scorer(messages)
    keyword_scores = bm25.score(query)

    # 2. Semantic similarity scores
    semantic = SemanticScorer(messages)
    semantic_scores = semantic.score(query)

    # 3. Recency decay scores
    recency_scores = score_recency(len(messages), lambda_decay)

    # 4. Landmark detection
    landmark_results = detect_landmarks(messages)

    # Combine signals
    results = []
    for i in range(len(messages)):
        # Landmark boost: 1.0 for normal messages, LANDMARK_BOOST for protected
        is_landmark = landmark_results[i].protection != ProtectionLevel.NONE
        boost = LANDMARK_BOOST if is_landmark else 1.0

        # Weighted sum with landmark boost applied to final score
        raw_score = (
            weights.keyword * keyword_scores[i]
            + weights.semantic * semantic_scores[i]
            + weights.recency * recency_scores[i]
        )

        # Apply landmark boost multiplicatively
        final_score = raw_score * boost

        # Clamp to [0, 1] — boost can push above 1.0
        final_score = min(1.0, final_score)

        results.append(MessageScore(
            index=i,
            keyword_score=keyword_scores[i],
            semantic_score=semantic_scores[i],
            recency_score=recency_scores[i],
            landmark_boost=boost,
            final_score=final_score,
            protection=landmark_results[i].protection,
        ))

    return results


def get_weights_for_query_type(query_type: str) -> ScoringWeights:
    """
    Return appropriate weights for a query type.

    Args:
        query_type: One of 'factual', 'analytical', 'procedural'.

    Returns:
        ScoringWeights configured for the query type.
    """
    mapping = {
        "factual": FACTUAL_WEIGHTS,
        "analytical": ANALYTICAL_WEIGHTS,
        "procedural": PROCEDURAL_WEIGHTS,
    }
    return mapping.get(query_type, ScoringWeights())
