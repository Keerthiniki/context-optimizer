from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(
    title="Context Optimizer",
    description=(
        "Intelligent context optimisation for multi-turn LLM conversations. "
        "Reduces token count by 40-60% while preserving answer quality through "
        "relevance scoring, landmark detection, and LLM-powered compression."
    ),
    version="1.0.0",
)

app.include_router(router, tags=["optimize"])


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "context-optimizer"}
