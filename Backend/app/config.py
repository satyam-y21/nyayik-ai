import os

from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Redirect HF cache to project dir to avoid filling C: drive
os.environ["HF_HOME"] = os.getenv(
    "HF_HOME",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../.cache/huggingface")
    ),
)


class Settings:
    PROJECT_NAME: str = "Nyayik AI: Indian Legal RAG Assistant"
    API_V1_STR: str = "/api"

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")

    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"
    )

    VECTOR_DB_PROVIDER: str = os.getenv("VECTOR_DB_PROVIDER", "faiss")
    VECTOR_DB_DIR: str = os.getenv("VECTOR_DB_DIR", "./Vectorstore/faiss_index")
    DATA_RAW_DIR: str = os.getenv("DATA_RAW_DIR", "./Data/raw")
    DATA_PROCESSED_DIR: str = os.getenv("DATA_PROCESSED_DIR", "./Data/processed")

    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "legal_documents")

    RETRIEVAL_SCORE_THRESHOLD: float = float(
        os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.65")
    )

    ENABLE_PERFORMANCE_LOGGING: bool = os.getenv(
        "ENABLE_PERFORMANCE_LOGGING", "true"
    ).lower() in ("true", "1", "yes")
    PREWARM_EMBEDDINGS: bool = os.getenv("PREWARM_EMBEDDINGS", "true").lower() in (
        "true",
        "1",
        "yes",
    )
    LOG_QUERY_TEXT: bool = os.getenv("LOG_QUERY_TEXT", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    PERFORMANCE_LOG_DIR: str = os.getenv("PERFORMANCE_LOG_DIR", "./Backend/logs")


settings = Settings()
