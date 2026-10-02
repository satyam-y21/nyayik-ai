import json
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from Backend.app.config import settings

# Indian Standard Time (UTC+05:30)
IST = timezone(timedelta(hours=5, minutes=30))


class PerformanceMetricsLogger:
    """Appends structured metrics to performance.jsonl. Thread-safe."""

    def __init__(self, log_dir: Optional[str] = None, enabled: Optional[bool] = None):
        self.enabled = (
            settings.ENABLE_PERFORMANCE_LOGGING if enabled is None else enabled
        )
        self.log_dir = Path(log_dir or settings.PERFORMANCE_LOG_DIR)
        self.log_file = self.log_dir / "performance.jsonl"
        self._lock = threading.Lock()
        self._initialized = False

    def _ensure_dir(self):
        if not self._initialized:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                self._initialized = True
            except Exception as e:
                print(
                    f"[WARNING] Failed to create metrics log directory {self.log_dir}: {e}"
                )

    def log_request(self, record: Dict[str, Any]) -> bool:
        """Write one request record. Swallows errors so requests never fail."""
        if not self.enabled:
            return False

        try:
            self._ensure_dir()

            # Format record defaults
            log_data = {
                "timestamp": record.get("timestamp", datetime.now(IST).isoformat()),
                "request_id": record.get("request_id", str(uuid.uuid4())),
                "user_query": record.get("user_query") or record.get("query_text"),
                "response_text": record.get("response_text"),
                "intent": record.get("intent", "UNKNOWN"),
                "execution_path": record.get("execution_path", "UNKNOWN"),
                "query_length": record.get("query_length", 0),
                "intent_latency_ms": record.get("intent_latency_ms"),
                "embedding_latency_ms": record.get("embedding_latency_ms"),
                "qdrant_latency_ms": record.get("qdrant_latency_ms"),
                "llm_latency_ms": record.get("llm_latency_ms"),
                "total_latency_ms": record.get("total_latency_ms"),
                "chunks_retrieved": record.get("chunks_retrieved"),
                "chunks_after_threshold": record.get("chunks_after_threshold"),
                "top_similarity_score": record.get("top_similarity_score"),
                "lowest_selected_similarity_score": record.get(
                    "lowest_selected_similarity_score"
                ),
                "retrieval_threshold": record.get(
                    "retrieval_threshold", settings.RETRIEVAL_SCORE_THRESHOLD
                ),
                "document_filter_applied": record.get("document_filter_applied", False),
                "identified_document_filter": record.get("identified_document_filter"),
                "response_mode": record.get("response_mode", "UNKNOWN"),
                "retrieval_attempted": record.get("retrieval_attempted", False),
                "retrieval_accepted": record.get("retrieval_accepted", False),
                "fallback_reason": record.get("fallback_reason"),
                "answer_generated": record.get("answer_generated", False),
                "fallback_triggered": record.get("fallback_triggered", False),
                "ambiguity_triggered": record.get("ambiguity_triggered", False),
                "error": record.get("error", False),
                "error_type": record.get("error_type"),
            }

            json_line = json.dumps(log_data)

            with self._lock:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json_line + "\n")
            return True

        except Exception as e:
            print(f"Metrics write failed: {e}")
            return False


_global_metrics_logger: Optional[PerformanceMetricsLogger] = None


def get_metrics_logger() -> PerformanceMetricsLogger:
    global _global_metrics_logger
    if _global_metrics_logger is None:
        _global_metrics_logger = PerformanceMetricsLogger()
    return _global_metrics_logger
