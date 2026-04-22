# AI Usage Note
## LEC AI Assignment 3

### Tools Used

Claude (via claude.ai) as a coding assistant throughout the project. No other AI tools were used.

### How I Used AI

I used Claude the same way I'd use it in a production engineering workflow — as a fast pair programmer that I directed, reviewed, and corrected throughout.

**Code acceleration:** I designed each module's interface, data structures, and algorithmic approach, then used Claude to generate initial implementations faster than typing them from scratch. Every generated file was reviewed, tested, and in several cases substantially rewritten when the output didn't match my actual interfaces or missed edge cases I'd specified.

**Debugging partner:** When tests failed or components didn't integrate cleanly, I shared error output with Claude and worked through fixes collaboratively. The pattern was always: I diagnose the problem, Claude suggests a fix, I verify the fix makes sense before applying it.

**Documentation drafting:** I outlined each document's structure and key points, then used Claude to draft prose from those outlines. I edited for accuracy, tone, and to ensure all technical claims match the real implementation and eval numbers.

### What Was Entirely My Work

**All architectural decisions** — the pipeline pattern, the four-signal scoring formula, adaptive weights per query type, selective summarisation strategy, and the confidence-based fallback. These came from my experience building Memory AgentCore, where I implemented similar relevance scoring and recency decay for multi-agent memory retrieval.

**The iterative improvement cycle** — running evals, diagnosing that analytical queries were the weak point (-0.50 delta), deciding the sequence of fixes (broader context for analytical → confidence fallback → high-detail protection → selective compression → threshold tuning), and knowing when to stop optimising. This diagnostic loop was the core engineering work.

**Evaluation design** — 10 conversations × 4 query types, LLM-as-judge rubric, per-type breakdown that made the analytical weakness visible, and the decision to prioritise quality parity over maximum compression.

**Every trade-off judgment** — MiniLM over OpenAI embeddings (cost vs accuracy), BM25 over TF-IDF (length normalisation), Haiku over local Llama (quality vs cost), stateless over Redis (simplicity vs persistence). I can defend each of these from first principles.

### How I Verified Quality

Every module has independent unit tests (225 total). The evaluation framework provided the ground truth — initial results showed -0.38 quality delta and 67% reduction. Over four improvement rounds I brought that to -0.05 delta and 44.4% reduction, with each round driven by per-query-type analysis of where the system was failing.

### Summary

AI made me faster at the parts that don't require judgment — boilerplate, test scaffolding, prose drafting. The engineering decisions — what to build, why this approach over alternatives, how to interpret results, and what to fix — were mine throughout. I'd use the same workflow on day one at LEC AI.
