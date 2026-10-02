import torch
from qdrant_client import QdrantClient

from Backend.app.config import settings


class LifecycleManager:
    """Singleton holders for embeddings and qdrant client."""

    _embeddings = None
    _qdrant_client = None

    @classmethod
    def get_embeddings(cls):
        """Get or create the embedding model. Warms CUDA on first call."""
        if cls._embeddings is None:
            provider = settings.EMBEDDING_PROVIDER.lower()
            if provider == "local":
                from langchain_community.embeddings import HuggingFaceEmbeddings

                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Loading embeddings on {device.upper()}")
                cls._embeddings = HuggingFaceEmbeddings(
                    model_name=settings.EMBEDDING_MODEL_NAME,
                    model_kwargs={"device": device},
                )
            elif provider == "openai":
                from langchain_openai import OpenAIEmbeddings

                cls._embeddings = OpenAIEmbeddings(
                    model=settings.EMBEDDING_MODEL_NAME
                )
            elif provider == "gemini":
                from langchain_google_genai import GoogleGenaiEmbeddings

                cls._embeddings = GoogleGenaiEmbeddings(
                    model=settings.EMBEDDING_MODEL_NAME
                )
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")

            # dummy inference to prime CUDA context
            try:
                _ = cls._embeddings.embed_query("warmup")
                print("Embedding model warmed up.")
            except Exception as w_err:
                print(f"Pre-warm skipped: {w_err}")

        return cls._embeddings

    @classmethod
    def get_qdrant_client(cls) -> QdrantClient:
        """Get or create the Qdrant connection."""
        if cls._qdrant_client is None:
            url = settings.QDRANT_URL
            api_key = settings.QDRANT_API_KEY
            if not url or "your_qdrant" in url:
                print("Qdrant URL not configured, using in-memory client")
                cls._qdrant_client = QdrantClient(":memory:")
            else:
                print(f"Connecting to Qdrant at {url}")
                cls._qdrant_client = QdrantClient(
                    url=url, api_key=api_key, timeout=120.0
                )
        return cls._qdrant_client
