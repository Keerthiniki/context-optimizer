# Context Optimizer — Integration Guide for Multi-Agent Platforms

## What Was Built

Standalone context optimisation pipeline for long-running agent sessions. Reduces context window size by 44% while maintaining answer quality parity — directly addressing context rot in multi-agent workflows where tool outputs, code edits, and bash results accumulate across missions.

### 1. Query Classifier (`query_classifier.py`)
**Technology:** Keyword regex with confidence scoring
- Classifies intent as factual / analytical / procedural
- Drives adaptive scoring weights downstream
- Low-confidence fallback expands context automatically

**Key Features:**
- Priority-ordered pattern matching (procedural → analytical → factual)
- Confidence signal triggers safety nets when uncertain
- Zero latency — pure regex, no LLM call

---

### 2. Landmark Detector (`landmark_detector.py`)
**Technology:** Rule-based pattern matching
- Flags decisions, commitments, action items, deadlines
- Detects tool-call chains (tool_use → tool_result pairs)
- PROTECTED messages never dropped or compressed

**Key Features:**
- Decisions: "we decided", "agreed on", "let's go with"
- Commitments: "I'll handle", "taking ownership"
- Tool chains: automatic pairing of tool_use + tool_result
- Sub-1ms execution time

---

### 3. Relevance Scorer (`relevance_scorer.py`)
**Technology:** Hybrid retrieval (BM25 + semantic + recency + landmark)
- Four complementary signals combined with adaptive weights
- BM25-ranked retrieval over conversation history (same approach as ctx_search)
- MiniLM-L6-v2 cosine similarity for paraphrase detection
- Exponential recency decay (λ=0.1)
- 2x multiplicative landmark boost

**Scoring Formula:**
```
score = α·BM25 + β·semantic + γ·recency × landmark_boost
```

**Adaptive Weights:**
| Signal | Factual | Analytical | Procedural |
|--------|---------|------------|------------|
| α keyword | 0.3 | 0.2 | 0.2 |
| β semantic | 0.2 | 0.3 | 0.3 |
| γ recency | 0.2 | 0.2 | 0.3 |
| δ landmark | 0.3 | 0.3 | 0.2 |

---

### 4. Selector (`message_selector.py`)
**Technology:** Priority-based selection with safety nets
- Strict priority order: landmarks → tool chains → tail → score threshold
- Adaptive thresholds per query type
- Four safety nets prevent catastrophic context loss

**Safety Nets:**
| Net | What It Catches |
|-----|----------------|
| Pair Preservation | Kept question always has at least a compressed answer |
| Quality Guardrail | Minimum 35–38% token retention enforced |
| Confidence Fallback | Low-confidence queries get expanded context |
| High-Detail Protection | Numbers, configs, code promoted from COMPRESS → KEEP |

---

### 5. Compressor (`compressor.py`)
**Technology:** Claude Haiku with selective prompts + caching
- Content-type-specific compression intensity
- SHA256 content-hash caching (zero-cost repeat compressions)
- Cost-aware skip (Haiku cost vs Sonnet savings check)

**Compression Profiles:**
| Content Type | Detection | Output |
|-------------|-----------|--------|
| Small talk | "ok", "thanks", short msgs | ≤15 words |
| Reasoning | "because", "however", "trade-off" | ≤60 words, logic preserved |
| Technical | code, configs, $amounts | ≤80 words, all specifics kept |

**Cost Model:**
- Haiku compression: ~$0.00043/call
- Downstream Sonnet saving: ~$0.0045/query
- Net ROI: ~10×
- Cost-aware skip below ~200 tokens

---

### 6. Assembler + Validator (`assembler.py`, `thread_validator.py`)
**Technology:** Thread reconstruction + structural checks
- Enforces user/assistant role alternation
- Injects [SUMMARY] markers for compressed clusters
- Tool-chain integrity verification
- Validation: no orphaned turns, valid roles, tool_result follows tool_use

---

## File Structure

```
context-optimizer/
├── src/
│   ├── classifier/
│   │   └── query_classifier.py        # Intent detection with confidence
│   ├── detector/
│   │   └── landmark_detector.py       # Decision/commitment/tool-chain detection
│   ├── scorer/
│   │   ├── bm25_scorer.py             # BM25 keyword scoring
│   │   ├── semantic_scorer.py         # MiniLM cosine similarity
│   │   ├── recency_scorer.py          # Exponential decay
│   │   └── relevance_scorer.py        # Combines all four signals
│   ├── selector/
│   │   └── message_selector.py        # Priority selection + safety nets
│   ├── compressor/
│   │   └── compressor.py              # Haiku compression + cache
│   ├── assembler/
│   │   └── assembler.py               # Thread reconstruction
│   ├── validator/
│   │   └── thread_validator.py        # Structural integrity
│   └── api/
│       ├── main.py                    # FastAPI app
│       ├── models.py                  # Pydantic request/response
│       └── routes.py                  # Pipeline orchestration
├── eval/
│   ├── conversations/                 # 10 synthetic conversations
│   ├── queries/                       # Multi-step queries
│   └── run_eval.py                    # LLM-as-judge evaluation
├── tests/                             # 225 unit + integration tests
└── docs/
    ├── architecture.md                # Design decisions + diagrams
    ├── report.md                      # Written report
    ├── roadmap.md                     # What I'd ship next
    └── ai_usage.md                    # AI usage note
```

---

## API Endpoints

### 1. Single Optimisation
```
POST /optimize
Request:
{
  "messages": [...],         # 50+ message conversation
  "query": "current query",
  "threshold_high": 0.4,    # optional
  "threshold_low": 0.2,     # optional
  "skip_compression": false  # optional — dry run mode
}

Response:
{
  "optimized_messages": [...],
  "metrics": {
    "token_count_original": 3109,
    "token_count_optimized": 1904,
    "token_reduction_percent": 38.76,
    "landmarks_preserved": 12,
    "tool_chains_preserved": 2,
    "messages_kept": 29,
    "messages_compressed": 29,
    "messages_dropped": 23,
    "compression_cost_usd": 0.000641,
    "assembly_latency_ms": 5140,
    "thread_valid": true
  },
  "score_breakdown": [...]    # per-message scores and actions
}
```

### 2. Batch Optimisation (Evaluation)
```
POST /optimize/batch
Request:
{
  "requests": [
    { "messages": [...], "query": "..." },
    { "messages": [...], "query": "..." }
  ]
}

Response:
{
  "results": [...],
  "aggregate_metrics": {
    "total_requests": 40,
    "avg_token_reduction_percent": 44.41,
    "total_compression_cost_usd": 0.016591,
    "avg_assembly_latency_ms": 5139.65,
    "all_threads_valid": true
  }
}
```

---

## Multi-Agent Integration Points

### Where It Fits in an Agent Platform

```
Agent receives mission
    ↓
Agent executes tool calls (bash, file read, code edit, etc.)
    ↓
Conversation history grows (tool_use + tool_result pairs)
    ↓
Before next Claude API call:
    ↓
┌─────────────────────────────────────────┐
│  Context Optimizer Pipeline              │
│  • Classify agent's likely intent        │
│  • Preserve tool chains + decisions      │
│  • Compress verbose tool outputs         │
│  • Assemble clean context window         │
└─────────────────────────────────────────┘
    ↓
Agent continues with optimised context
    ↓
Lower cost, better accuracy, faster response
```

### SDK Engine Integration

The optimizer slots into the conversation assembly layer — the point where message history is prepared before the next Claude API call:

```python
# Before (full context)
messages = session.get_full_history()
response = await claude.messages.create(messages=messages)

# After (optimised context)
messages = session.get_full_history()
if count_tokens(messages) > OPTIMIZATION_THRESHOLD:
    result = optimizer.optimize(messages=messages, query=current_intent)
    messages = result.optimized_messages
response = await claude.messages.create(messages=messages)
```

Configurable threshold — only optimises when context exceeds N tokens. Short sessions pass through unchanged.

### What Gets Preserved

| Content Type | Treatment | Why |
|-------------|-----------|-----|
| Decisions / commitments | Always KEEP | Downstream agents need upstream decisions |
| Tool chains (tool_use + tool_result) | Always KEEP | Breaking chains creates invalid context |
| Recent messages (last 5) | Always KEEP | Immediate context for next action |
| Sub-mission references | Always KEEP | Dependency chain integrity |
| Verbose bash/file output | COMPRESS | 315KB output → 2-sentence summary |
| Acknowledgements / filler | DROP | Zero information value |

### Dependency Chain Awareness

When agents create sub-missions with dependency chains, decisions from upstream agents must carry forward to downstream agents. The landmark detector catches these automatically — "we decided", "the approach is", "confirmed that" — and flags them as PROTECTED. This means an agent dispatched after a dependency is satisfied receives the upstream decision verbatim, not a compressed approximation.

### Cost Impact Per Mission

```
Without optimizer:
  10 tool calls × 2 messages each = 20 messages
  + prior context = ~50 messages
  × 3 subsequent Claude calls = 150 messages processed
  Cost: ~$0.045 in context tokens alone

With optimizer (at 44% reduction):
  50 messages → 28 messages per call
  × 3 calls = 84 messages processed
  Compression cost: ~$0.001
  Net saving: ~$0.019 per mission (42% reduction)

At 100 missions/day: ~$1.90/day saved
At 1000 missions/day: ~$19/day saved
```

### Session Resume Compatibility

When an agent session fails and needs resume, the optimizer preserves the full conversation state on the first call. On subsequent calls within the resumed session, it optimises accumulated history while keeping the resume context intact. The confidence fallback ensures resumed sessions — which often have unusual context patterns — get expanded rather than compressed.

---

## Evaluation Results

| Metric | Result |
|--------|--------|
| Token reduction | **44.4%** |
| Quality delta | **-0.05** (parity) |
| Wins / ties / losses | 9 / 21 / 10 |
| All threads valid | ✅ |
| Compression cost (40 queries) | $0.017 |

### By Query Type
| Type | Reduction | Quality Δ |
|------|-----------|-----------|
| Factual | 45.0% | +0.2 |
| Analytical | 44.6% | -0.2 |
| Procedural | 46.0% | -0.1 |
| Landmark | 42.0% | -0.1 |

---

## Technology Stack

### Models Used
| Component | Model | Why |
|-----------|-------|-----|
| Compression | Claude Haiku 4.5 | Cheapest, same stack, quality sufficient |
| Eval Judge | Claude Sonnet 4.6 | Better reasoning for quality scoring |
| Embeddings | MiniLM-L6-v2 | Free, local, 20ms batch encode |

### Dependencies
- FastAPI + Pydantic — API + validation
- sentence-transformers — local embeddings
- rank-bm25 — keyword scoring
- tiktoken — token counting
- anthropic — Haiku API client

### No External Infrastructure Required
- No Redis (in-memory cache, Redis-ready)
- No PostgreSQL (stateless per-request)
- No message broker (synchronous pipeline)
- Single `pip install` to run

---

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Query classification | <1ms | Pure regex |
| Landmark detection | <1ms | Pure regex |
| BM25 scoring | ~10ms | Rebuilds per request |
| MiniLM encoding | ~20-40ms | Batch encode, CPU |
| Selection + safety nets | ~5ms | Algorithmic |
| Haiku compression | 300-1500ms | Per cluster, cached after first call |
| Assembly + validation | ~5ms | Algorithmic |
| **Full pipeline (no cache)** | **~5s** | **Dominated by Haiku calls** |
| **Full pipeline (cached)** | **<100ms** | **All local** |

---

## Backward Compatibility

 **Fully standalone** — no changes to existing platform required
- Stateless REST API — call it or don't
- No schema changes to any database
- No new infrastructure dependencies
- Configurable threshold — skip optimisation for short sessions
- Graceful degradation — if Haiku API fails, local fallback maintains service

---

## Future Enhancements

| Feature | Effort | Impact |
|---------|--------|--------|
| SSE streaming of optimised context | 2-3 days | Reduced perceived latency |
| Redis caching layer | Half day | Cross-instance compression cache |
| User feedback loop | 3-4 days | Automatic weight tuning from production data |
| Domain-specific landmark patterns | 1-2 days | Custom vocabularies per team/project |
| Multilingual support (XLM-RoBERTa) | 3-4 days | Non-English agent sessions |
| Per-mission cost tracking integration | 1 day | Optimizer cost in mission reports |

---

## Ready to Integrate

The optimizer runs as a standalone service:
1. `pip install -r requirements.txt`
2. Set `ANTHROPIC_API_KEY` in `.env`
3. `uvicorn src.api.main:app --reload`
4. Call `POST /optimize` from the SDK engine layer when context exceeds threshold

No changes to agent logic, mission structure, or dispatch flow required.
