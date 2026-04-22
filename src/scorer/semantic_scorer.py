import numpy as np
from sentence_transformers import SentenceTransformer

# Load model once at module level — ~2s cold start, then reused.
_MODEL = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the model on first use."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def _extract_text(message: dict) -> str:
    """Extract searchable text from a message."""
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


class SemanticScorer:
    """
    Scores conversation messages against a query using cosine similarity
    on MiniLM embeddings.

    Usage:
        scorer = SemanticScorer(messages)
        scores = scorer.score(query)
        # scores[i] = semantic similarity of messages[i] to query
    """

    def __init__(self, messages: list[dict]):
        """
        Batch encode all messages on init.

        Args:
            messages: List of conversation messages with 'role' and 'content'.
        """
        self._messages = messages
        self._model = _get_model()

        texts = [_extract_text(m) for m in messages]

        # Batch encode all messages in one call — much faster than one-by-one.
        # Replace empty strings with a placeholder to avoid encoding errors.
        texts_clean = [t if t.strip() else "[empty]" for t in texts]

        if texts_clean:
            self._embeddings = self._model.encode(
                texts_clean,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,  # Pre-normalize for fast cosine sim
            )
        else:
            self._embeddings = np.array([])

    def score(self, query: str) -> list[float]:
        """
        Score all messages against the query via cosine similarity.

        Args:
            query: User's current query string.

        Returns:
            List of float scores in [0, 1], one per message.
        """
        if len(self._messages) == 0:
            return []

        if not query.strip():
            return [0.0] * len(self._messages)

        # Encode query with same normalization
        query_embedding = self._model.encode(
            [query],
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]

        # Cosine similarity = dot product when both are normalized
        similarities = self._embeddings @ query_embedding

        # Cosine similarity ranges [-1, 1]. Clamp to [0, 1] since
        # negative similarity means unrelated — treat as zero.
        scores = [max(0.0, float(s)) for s in similarities]

        # Normalize so max = 1.0
        max_score = max(scores) if scores else 0.0
        if max_score > 0:
            return [s / max_score for s in scores]
        return [0.0] * len(self._messages)

    def get_top_k(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """
        Return top-k message indices by semantic similarity.

        Args:
            query: User's current query string.
            k: Number of top results to return.

        Returns:
            List of (message_index, score) tuples, sorted descending.
        """
        scores = self.score(query)
        indexed = [(i, s) for i, s in enumerate(scores)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:k]
