from rank_bm25 import BM25Okapi
import re


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer.
    Lowercases, strips punctuation, removes single-char tokens.
    Good enough for BM25 on conversational English — no need for
    NLTK or spaCy overhead.
    """
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return [t for t in tokens if len(t) > 1]


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


class BM25Scorer:
    """
    Scores conversation messages against a query using BM25.

    Usage:
        scorer = BM25Scorer(messages)
        scores = scorer.score(query)
        # scores[i] = BM25 relevance of messages[i] to query
    """

    def __init__(self, messages: list[dict]):
        """
        Build BM25 index from conversation messages.

        Args:
            messages: List of conversation messages with 'role' and 'content'.
        """
        self._messages = messages
        self._corpus = [_tokenize(_extract_text(m)) for m in messages]

        # Handle edge case: all empty messages
        if all(len(doc) == 0 for doc in self._corpus):
            self._bm25 = None
        else:
            self._bm25 = BM25Okapi(self._corpus)

    def score(self, query: str) -> list[float]:
        """
        Score all messages against the query.

        Args:
            query: User's current query string.

        Returns:
            List of float scores, one per message, in same order as input.
            Scores are normalized to [0, 1] range.
        """
        if self._bm25 is None:
            return [0.0] * len(self._messages)

        query_tokens = _tokenize(query)
        if not query_tokens:
            return [0.0] * len(self._messages)

        raw_scores = self._bm25.get_scores(query_tokens)

        # BM25 IDF degrades with very small corpora (1-2 docs) — terms that
        # appear in every document get IDF=0, producing all-zero scores.
        # Fallback: simple token overlap ratio when BM25 returns all zeros.
        max_score = max(raw_scores) if len(raw_scores) > 0 else 0.0
        if max_score <= 0.0 and len(self._messages) <= 2:
            return self._fallback_overlap_scores(query_tokens)

        if max_score > 0:
            return [s / max_score for s in raw_scores]
        return [0.0] * len(self._messages)

    def _fallback_overlap_scores(self, query_tokens: list[str]) -> list[float]:
        """
        Token overlap fallback for tiny corpora where BM25 IDF breaks down.
        Scores = proportion of query tokens found in each message.
        """
        query_set = set(query_tokens)
        if not query_set:
            return [0.0] * len(self._messages)

        scores = []
        for doc_tokens in self._corpus:
            doc_set = set(doc_tokens)
            overlap = len(query_set & doc_set) / len(query_set)
            scores.append(overlap)

        max_s = max(scores) if scores else 0.0
        if max_s > 0:
            return [s / max_s for s in scores]
        return [0.0] * len(self._messages)

    def get_top_k(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """
        Return top-k message indices by BM25 score.

        Args:
            query: User's current query string.
            k: Number of top results to return.

        Returns:
            List of (message_index, score) tuples, sorted by score descending.
        """
        scores = self.score(query)
        indexed = [(i, s) for i, s in enumerate(scores)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:k]
