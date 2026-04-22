import argparse
import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict

from anthropic import Anthropic
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.routes import run_pipeline
from src.api.models import OptimizeRequest

load_dotenv()

client = Anthropic()
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")
ANSWER_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")

CONVERSATIONS_DIR = Path(__file__).parent / "conversations"
QUERIES_DIR = Path(__file__).parent / "queries"
RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Result for a single query evaluation."""
    conversation_id: str
    query: str
    query_type: str
    # Metrics
    original_token_count: int = 0
    optimized_token_count: int = 0
    token_reduction_percent: float = 0.0
    pipeline_latency_ms: float = 0.0
    compression_cost_usd: float = 0.0
    # Quality scores (1-10)
    full_context_score: float = 0.0
    optimized_context_score: float = 0.0
    quality_delta: float = 0.0  # optimised - full (positive = better)
    # Pipeline stats
    messages_kept: int = 0
    messages_compressed: int = 0
    messages_dropped: int = 0
    landmarks_preserved: int = 0
    thread_valid: bool = True
    # Answers (for inspection)
    full_context_answer: str = ""
    optimized_context_answer: str = ""
    judge_reasoning: str = ""


@dataclass
class EvalSummary:
    """Aggregate evaluation results."""
    total_queries: int = 0
    avg_token_reduction_percent: float = 0.0
    avg_full_context_score: float = 0.0
    avg_optimized_context_score: float = 0.0
    avg_quality_delta: float = 0.0
    optimized_wins: int = 0
    optimized_ties: int = 0
    optimized_losses: int = 0
    avg_pipeline_latency_ms: float = 0.0
    total_compression_cost_usd: float = 0.0
    all_threads_valid: bool = True
    # Per query type breakdown
    by_query_type: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question based ONLY
on the conversation history provided. Be specific and reference details from the conversation.
Keep your answer concise — 2-4 sentences maximum."""


def get_answer(messages: list[dict], query: str, dry_run: bool = False) -> str:
    """Get an answer to the query using the provided context."""
    if dry_run:
        return "[DRY RUN] Answer would be generated here."

    # Build the prompt: conversation history + query
    context_str = ""
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            content = " ".join(text_parts)
        context_str += f"[{role}]: {content}\n"

    response = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=300,
        system=ANSWER_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"CONVERSATION HISTORY:\n{context_str}\n\nQUESTION: {query}",
        }],
    )

    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating answer quality.

You will be given:
- A conversation query
- Answer A (from full context)
- Answer B (from optimised context)

Score EACH answer from 1-10 on these criteria:
- Factual accuracy (does it match what was discussed?)
- Completeness (does it cover all relevant points?)
- Relevance (does it directly address the query?)

Respond with ONLY valid JSON, no markdown fences:
{
  "score_a": <1-10>,
  "score_b": <1-10>,
  "reasoning": "<2-3 sentences explaining your scoring>"
}"""


def judge_answers(
    query: str,
    answer_full: str,
    answer_optimized: str,
    dry_run: bool = False,
) -> tuple[float, float, str]:
    """
    Judge both answers and return (score_full, score_optimized, reasoning).
    """
    if dry_run:
        return 7.0, 7.0, "[DRY RUN] Judging would happen here."

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=300,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"QUERY: {query}\n\n"
                f"ANSWER A (full context):\n{answer_full}\n\n"
                f"ANSWER B (optimised context):\n{answer_optimized}"
            ),
        }],
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        result = json.loads(raw)
        return (
            float(result.get("score_a", 5)),
            float(result.get("score_b", 5)),
            result.get("reasoning", "No reasoning provided."),
        )
    except (json.JSONDecodeError, ValueError):
        print(f"    ⚠ Judge returned invalid JSON, using defaults")
        return 5.0, 5.0, f"Parse error. Raw: {raw[:200]}"


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def load_conversations() -> dict[str, dict]:
    """Load all generated conversations."""
    conversations = {}
    for path in sorted(CONVERSATIONS_DIR.glob("*.json")):
        with open(path) as f:
            conv = json.load(f)
        conversations[conv["id"]] = conv
    return conversations


def load_queries() -> list[dict]:
    """Load all query sets."""
    queries_path = QUERIES_DIR / "queries.json"
    with open(queries_path) as f:
        return json.load(f)


def run_eval(dry_run: bool = False):
    """Run full evaluation."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    conversations = load_conversations()
    query_sets = load_queries()

    print(f"Loaded {len(conversations)} conversations, {len(query_sets)} query sets")
    if dry_run:
        print("🏃 DRY RUN MODE — no API calls for answers/judging\n")
    else:
        print()

    all_results: list[QueryResult] = []
    total_queries = sum(len(qs.get("queries", [])) for qs in query_sets)
    current = 0

    for query_set in query_sets:
        conv_id = query_set["conversation_id"]
        conv = conversations.get(conv_id)

        if conv is None:
            print(f"⚠ Conversation {conv_id} not found, skipping")
            continue

        messages = conv["messages"]
        print(f"{'='*60}")
        print(f"Conversation: {conv_id} ({len(messages)} messages)")
        print(f"{'='*60}")

        for q in query_set.get("queries", []):
            current += 1
            query = q["query"]
            query_type = q.get("type", "unknown")

            print(f"\n  [{current}/{total_queries}] ({query_type}) {query[:60]}...")

            # --- Step 1: Answer with FULL context ---
            print(f"    Getting full-context answer...", end=" ", flush=True)
            full_answer = get_answer(messages, query, dry_run=dry_run)
            print("✓")

            # --- Step 2: Run optimizer pipeline ---
            print(f"    Running optimizer pipeline...", end=" ", flush=True)
            opt_request = OptimizeRequest(
                messages=messages,
                query=query,
                skip_compression=dry_run,  # skip Haiku in dry run
            )

            try:
                pipeline_result = run_pipeline(opt_request)
                print("✓")
            except Exception as e:
                print(f"✗ Pipeline error: {e}")
                continue

            optimized_messages = pipeline_result.optimized_messages
            metrics = pipeline_result.metrics

            # --- Step 3: Answer with OPTIMISED context ---
            print(f"    Getting optimised-context answer...", end=" ", flush=True)
            opt_answer = get_answer(optimized_messages, query, dry_run=dry_run)
            print("✓")

            # --- Step 4: Judge both answers ---
            print(f"    Judging answer quality...", end=" ", flush=True)
            score_full, score_opt, reasoning = judge_answers(
                query, full_answer, opt_answer, dry_run=dry_run,
            )
            print("✓")

            # --- Record result ---
            result = QueryResult(
                conversation_id=conv_id,
                query=query,
                query_type=query_type,
                original_token_count=metrics.token_count_original,
                optimized_token_count=metrics.token_count_optimized,
                token_reduction_percent=metrics.token_reduction_percent,
                pipeline_latency_ms=metrics.assembly_latency_ms,
                compression_cost_usd=metrics.compression_cost_usd,
                full_context_score=score_full,
                optimized_context_score=score_opt,
                quality_delta=round(score_opt - score_full, 2),
                messages_kept=metrics.messages_kept,
                messages_compressed=metrics.messages_compressed,
                messages_dropped=metrics.messages_dropped,
                landmarks_preserved=metrics.landmarks_preserved,
                thread_valid=metrics.thread_valid,
                full_context_answer=full_answer,
                optimized_context_answer=opt_answer,
                judge_reasoning=reasoning,
            )
            all_results.append(result)

            print(f"    → Tokens: {metrics.token_count_original} → {metrics.token_count_optimized} "
                  f"({metrics.token_reduction_percent}% reduction)")
            print(f"    → Quality: full={score_full}, optimised={score_opt}, "
                  f"delta={result.quality_delta:+.1f}")

    # --- Compute summary ---
    summary = compute_summary(all_results)

    # --- Save results ---
    results_output = {
        "summary": asdict(summary),
        "results": [asdict(r) for r in all_results],
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = "_dryrun" if dry_run else ""
    results_path = RESULTS_DIR / f"eval_{timestamp}{suffix}.json"

    with open(results_path, "w") as f:
        json.dump(results_output, f, indent=2)

    # --- Print summary ---
    print_summary(summary, results_path)

    return summary, all_results


def compute_summary(results: list[QueryResult]) -> EvalSummary:
    """Compute aggregate metrics from all query results."""
    if not results:
        return EvalSummary()

    n = len(results)
    summary = EvalSummary(
        total_queries=n,
        avg_token_reduction_percent=round(
            sum(r.token_reduction_percent for r in results) / n, 2
        ),
        avg_full_context_score=round(
            sum(r.full_context_score for r in results) / n, 2
        ),
        avg_optimized_context_score=round(
            sum(r.optimized_context_score for r in results) / n, 2
        ),
        avg_quality_delta=round(
            sum(r.quality_delta for r in results) / n, 2
        ),
        optimized_wins=sum(1 for r in results if r.quality_delta > 0),
        optimized_ties=sum(1 for r in results if r.quality_delta == 0),
        optimized_losses=sum(1 for r in results if r.quality_delta < 0),
        avg_pipeline_latency_ms=round(
            sum(r.pipeline_latency_ms for r in results) / n, 2
        ),
        total_compression_cost_usd=round(
            sum(r.compression_cost_usd for r in results), 6
        ),
        all_threads_valid=all(r.thread_valid for r in results),
    )

    # Per query type breakdown
    type_groups: dict[str, list[QueryResult]] = {}
    for r in results:
        type_groups.setdefault(r.query_type, []).append(r)

    for qtype, group in type_groups.items():
        gn = len(group)
        summary.by_query_type[qtype] = {
            "count": gn,
            "avg_token_reduction_percent": round(
                sum(r.token_reduction_percent for r in group) / gn, 2
            ),
            "avg_full_context_score": round(
                sum(r.full_context_score for r in group) / gn, 2
            ),
            "avg_optimized_context_score": round(
                sum(r.optimized_context_score for r in group) / gn, 2
            ),
            "avg_quality_delta": round(
                sum(r.quality_delta for r in group) / gn, 2
            ),
        }

    return summary


def print_summary(summary: EvalSummary, results_path: Path):
    """Print formatted summary to console."""
    print(f"\n\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}\n")

    print(f"  Queries evaluated:        {summary.total_queries}")
    print(f"  Avg token reduction:      {summary.avg_token_reduction_percent}%")
    print(f"  Avg full-context score:   {summary.avg_full_context_score}/10")
    print(f"  Avg optimised score:      {summary.avg_optimized_context_score}/10")
    print(f"  Avg quality delta:        {summary.avg_quality_delta:+.2f}")
    print(f"  Optimised wins/ties/loss: {summary.optimized_wins}/{summary.optimized_ties}/{summary.optimized_losses}")
    print(f"  Avg pipeline latency:     {summary.avg_pipeline_latency_ms}ms")
    print(f"  Total compression cost:   ${summary.total_compression_cost_usd:.4f}")
    print(f"  All threads valid:        {summary.all_threads_valid}")

    if summary.by_query_type:
        print(f"\n  BY QUERY TYPE:")
        for qtype, stats in summary.by_query_type.items():
            print(f"    {qtype}: "
                  f"reduction={stats['avg_token_reduction_percent']}%, "
                  f"full={stats['avg_full_context_score']}, "
                  f"opt={stats['avg_optimized_context_score']}, "
                  f"delta={stats['avg_quality_delta']:+.2f}")

    print(f"\n  Full results: {results_path}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Context Optimizer evaluation")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without making API calls for answers/judging",
    )
    args = parser.parse_args()

    run_eval(dry_run=args.dry_run)
