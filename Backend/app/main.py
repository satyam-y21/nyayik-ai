import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Backend.app.api.endpoints import router as api_router
from Backend.app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm models on startup so the first query isn't slow."""
    import time

    try:
        from Backend.app.utils.models import get_llm

        _ = get_llm()
        print("LLM singleton ready")
    except Exception as e:
        print(f"LLM pre-warm skipped: {e}")

    if settings.PREWARM_EMBEDDINGS or os.getenv(
        "PREWARM_EMBEDDINGS", "false"
    ).lower() in ("true", "1", "yes"):
        try:
            print("Pre-loading embeddings & vector store client...")
            from Backend.app.utils.lifecycle import LifecycleManager

            t0 = time.perf_counter()
            _ = LifecycleManager.get_embeddings()
            _ = LifecycleManager.get_qdrant_client()
            t_ms = (time.perf_counter() - t0) * 1000
            print(f"Warmup done in {t_ms:.0f}ms")
        except Exception as e:
            print(f"Embedding pre-warm skipped: {e}")
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Indian Legal RAG Assistant API",
    version="0.1.0",
    lifespan=lifespan,
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production/Docker needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
def read_root():
    return {"message": "Welcome to Nyayik AI: Indian Legal RAG Assistant API"}
