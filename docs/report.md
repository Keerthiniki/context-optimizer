# Written Report — Context Optimizer
## LEC AI Assignment 3

### What I Built

A multi-stage processing pipeline that takes a long conversation and a user query, then returns an optimised context window that is 44% smaller on average while maintaining answer quality parity with full context.

The system addresses context rot — the gradual degradation of response quality as irrelevant history crowds the context window. In production multi-agent systems, agents accumulate session history rapidly. A coding agent processing 10 tool calls per mission can hit 50+ context messages within a single mission. Without optimisation, you pay for noise and the model starts missing things. This pipeline sits between the conversation history and the next LLM call and fixes that.

The pipeline classifies the query type (factual, analytical, procedural), detects landmark messages (decisions, commitments, deadlines) that must never be lost, scores every message using four complementary signals (BM25 keyword matching, MiniLM semantic similarity, exponential recency decay, and a 2x landmark boost), then selects which messages to keep verbatim, compress via Claude Haiku, or drop entirely. Adaptive scoring weights and selection thresholds shift based on query type — factual queries prioritise precise keyword matches and landmarks, while analytical queries preserve broader context for coverage. A confidence-based fallback expands context when the system is uncertain about what the user needs.

The compressor uses content-type-specific prompts: small talk gets compressed aggressively into a single sentence, reasoning chains get light compression that preserves arguments and trade-offs, and technical content gets detailed preservation of numbers, configs, and steps. Cost-aware skip prevents compression when the Haiku API cost exceeds downstream Sonnet savings.

### Real Numbers

Evaluated across 10 synthetic conversations × 4 query types = 40 total queries:

- Token reduction: 44.4% average (target was 40–60%)
- Quality: optimised context scored 8.40/10 vs full context at 8.45/10 — a delta of -0.05 that falls within the noise range of LLM-as-judge evaluation
- 75% of queries scored equal or better than full context (9 wins, 21 ties, 10 losses)
- Factual queries improved with optimisation (+0.2 delta) — removing noise helps the model find specific facts
- All 40 assembled threads passed structural validation
- Compression cost across all 40 queries: $0.017

### What Broke

The hardest problem was balancing compression aggressiveness against answer quality. My first eval run achieved 67% reduction but scored -0.38 quality delta — too aggressive. Analytical queries were the worst at -0.50 because the system compressed reasoning chains and comparisons that summaries depend on.

The fix required multiple iterations. I lowered selection thresholds for analytical queries, added high-detail detection to prevent compressing messages with numbers and configs, built selective summarisation with different prompts per content type, and added a confidence-based fallback that expands context when the classifier has low confidence. Each change improved one dimension but risked regressing another — the final tuning pass required accepting that a -0.05 delta is a better outcome than chasing +0.10 at the cost of falling below 40% reduction.

The other significant failure was that my rule-based landmark detector misses ambiguous language. "Let's lock that in" is a commitment but doesn't match my regex patterns. A trained classifier would catch these, but there's no labelled dataset for landmark vs non-landmark conversation messages, and the rule-based approach is fast, explainable, and covers the clear-cut cases.

### What I Learnt

Semantic and keyword signals are genuinely complementary, not redundant. My first attempt used only semantic similarity, which worked for analytical queries but failed on factual queries like "what did we decide about the database" — the query and the decision message ("let's go with PostgreSQL") use completely different vocabulary, so cosine similarity scored the decision low. Adding BM25 keyword scoring fixed this immediately.

The evaluation framework was more valuable than I expected. Without LLM-as-judge scoring broken down by query type, I would have shipped a system that looked good on average but consistently failed on analytical queries. The per-type breakdown made the problem visible and directed every subsequent improvement.

Net cost is genuinely positive. At $0.00043 per compression call saving $0.0045 in downstream Sonnet tokens, the ROI is roughly 10x. The cost-aware skip ensures compression is never triggered when the math doesn't work — clusters below 50 tokens are summarised locally without an API call.

### Honest Limitations

The -0.05 quality delta means the system is at parity, not strictly better. On individual queries, 10 out of 40 scored lower than full context. Three of those scored -2.0 — significant enough that a user would notice. The system trades a small quality risk on complex analytical queries for 44% cost savings and reduced noise on everything else.

The evaluation uses synthetic conversations, which are cleaner and more structured than real conversations. Real agent sessions have more noise, code blocks, tool outputs, and ambiguity. Performance on production data would likely differ.

The LLM-as-judge uses the same model family (Claude) as the compression step, which introduces potential circularity. A more rigorous evaluation would use a different model family as judge.

### Production Application

This system was designed with multi-agent coding platforms in mind — specifically the problem of context management in long-running agent sessions where tool outputs, bash results, file reads, and code edits accumulate rapidly.

In a platform dispatching multiple agents per mission, each running in isolated worktrees with MCP tool access, the context window fills fast. A single agent processing 10 tool calls generates 20+ messages (tool_use + tool_result pairs) before any reasoning even begins. By mission 3 of a complex project, the accumulated context can easily hit 100+ messages — most of it stale tool output from earlier missions.

The optimizer addresses this at three levels. First, BM25-ranked retrieval over conversation history (the same approach used by context-mode's ctx_search) surfaces relevant prior context without shipping everything. Second, landmark preservation ensures decisions and commitments made in earlier missions carry forward — critical when agents create sub-missions with dependency chains, because a downstream agent needs to know what the upstream agent decided. Third, cost-aware compression means the optimiser only uses Haiku when the math works — if an agent session is short enough that full context is cheaper, compression gets skipped automatically.

The integration point would be the SDK engine layer, between conversation history assembly and the next Claude API call. When accumulated context exceeds a configurable token threshold, the optimizer runs — classifying the agent's likely intent, preserving tool chains and decision landmarks, compressing verbose outputs, and assembling a clean context. For sessions that are under the threshold, it passes through unchanged. This means agents that need full context keep it, while agents drowning in stale history get a focused, cheaper context window.

The net effect: lower per-mission cost, better agent accuracy on long sessions, and faster response times from reduced context size — all without changing agent logic or mission structure.
