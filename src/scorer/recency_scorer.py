import math
from dotenv import load_dotenv
import os

load_dotenv()

DEFAULT_LAMBDA = float(os.getenv("LAMBDA_DECAY", "0.1"))


def score_recency(
    num_messages: int,
    lambda_decay: float = DEFAULT_LAMBDA,
) -> list[float]:
    """
    Compute recency scores for all messages in a conversation.

    Args:
        num_messages: Total number of messages.
        lambda_decay: Decay rate. Higher = faster decay. Default from .env.

    Returns:
        List of float scores in [0, 1], one per message.
        Last message = 1.0, earlier messages decay exponentially.
    """
    if num_messages == 0:
        return []

    scores = []
    for i in range(num_messages):
        distance_from_end = (num_messages - 1) - i
        score = math.exp(-lambda_decay * distance_from_end)
        scores.append(score)

    return scores
