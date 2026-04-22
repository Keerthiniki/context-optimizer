# Roadmap — What I'd Ship Next
## LEC AI Assignment 3

### 1. Streaming Optimised Context via SSE

**Why:** Currently the full pipeline runs before returning anything. For long conversations (200+ messages), the user waits several seconds. Streaming assembled messages as they're confirmed would reduce perceived latency to near-zero for the first message.

**How:** Wrap the assembler output in a Server-Sent Events stream. Messages confirmed as KEEP can be streamed immediately. COMPRESS messages stream once Haiku returns. The client renders incrementally. This aligns directly with DevFleet's existing SSE architecture.

**Effort:** 2-3 days. FastAPI's StreamingResponse plus an async generator over the assembler output.

### 2. User Feedback Loop for Weight Tuning

**Why:** The current scoring weights were tuned against synthetic eval data. Production conversations will have different distributions of message types, vocabulary, and query patterns. Without feedback, the weights will drift from optimal.

**How:** Expose a POST /feedback endpoint that accepts a query ID and a binary signal (good/bad answer). Store signals in SQLite. Run a nightly job that adjusts scoring weights using Bayesian optimisation against accumulated feedback. Start with the current weights as priors.

**Effort:** 3-4 days. The feedback endpoint is trivial. The optimisation loop needs careful design to avoid catastrophic weight shifts — constrain updates to ±10% per cycle.

### 3. Domain-Specific Landmark Patterns

**Why:** The current landmark detector uses generic patterns ("we decided", "action item", "deadline"). Domain-specific conversations use different vocabulary — a legal team says "the parties agree", a sales team says "let's pencil that in", an engineering team says "let's lock that in". Missing these means dropping critical messages.

**How:** Expose a configuration endpoint where teams register custom landmark patterns. Store patterns per tenant in a JSON config. The landmark detector loads tenant patterns at request time and merges with the base patterns. No redeployment needed.

**Effort:** 1-2 days. The detector already uses compiled regex — extending it to accept additional patterns is straightforward.

### 4. Redis Caching Layer

**Why:** The current in-memory compression cache only persists within a single process run. In a multi-instance deployment behind a load balancer, each instance maintains its own cache — the same conversation compressed on instance A produces a cache miss on instance B.

**How:** Swap the Python dict for a Redis client. The cache key is already a content-addressable SHA256 hash, which works identically in Redis. Add a TTL of 24 hours to prevent unbounded cache growth. The change is literally one line in the Compressor constructor plus a Redis connection string in .env.

**Effort:** Half a day. The abstraction already exists — the cache interface is get/set by key.

### 5. Multilingual Support

**Why:** MiniLM-L6-v2 degrades significantly on non-English text. For teams operating in multiple languages, semantic similarity scores become unreliable, causing the selector to make poor keep/compress/drop decisions.

**How:** Swap MiniLM for XLM-RoBERTa-base, which handles 100+ languages with comparable English performance and acceptable latency. The semantic scorer interface stays the same — only the model name changes. BM25 keyword scoring needs language-aware tokenisation (use spaCy's multi-language tokenizer). Landmark patterns need per-language regex sets.

**Effort:** 3-4 days. Model swap is quick. Keyword tokenisation and landmark patterns for each target language are the real work.
