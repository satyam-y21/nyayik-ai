import os
import sys
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional


def percentiles(vals: List[float], p: float) -> float:
    """Calculate percentile from a list of sorted numbers."""
    if not vals:
        return 0.0
    sorted_vals = sorted(vals)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return d0 + d1


def analyze_performance_logs(log_file_path: Optional[str] = None):
    """Parses JSONL performance metrics log and prints a comprehensive analytical report."""
    if log_file_path is None:
        log_file_path = os.path.join(
            os.path.dirname(__file__), "../Backend/logs/performance.jsonl"
        )

    log_path = Path(log_file_path)

    print("\nNYAYA AI — RUNTIME PERFORMANCE REPORT")
    print("=====================================")
    print(f"Source Log: {log_path.resolve()}")

    if not log_path.exists():
        print("[NOTICE] No performance log file found at path.")
        print(
            "To generate performance logs, set ENABLE_PERFORMANCE_LOGGING=true and run queries against the backend."
        )
        return

    records: List[Dict[str, Any]] = []
    malformed_count = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(data)
            except json.JSONDecodeError:
                malformed_count += 1

    total_records = len(records)
    print(f"Total Logged Requests: {total_records}")
    if malformed_count > 0:
        print(f"[WARNING] Malformed Lines Ignored: {malformed_count}")

    if total_records == 0:
        print("No valid metric records available to analyze.")
        return

    # 1. Request Distribution
    intents: Dict[str, int] = {}
    paths: Dict[str, int] = {}

    for r in records:
        intent = r.get("intent", "UNKNOWN")
        path = r.get("execution_path", "UNKNOWN")
        intents[intent] = intents.get(intent, 0) + 1
        paths[path] = paths.get(path, 0) + 1

    print("\n--- REQUEST DISTRIBUTION ---")
    for intent, count in sorted(intents.items()):
        pct = (count / total_records) * 100
        print(f"  {intent:<20}: {count:>4} ({pct:>5.1f}%)")

    print("\n--- EXECUTION PATHS ---")
    for path, count in sorted(paths.items()):
        pct = (count / total_records) * 100
        print(f"  {path:<20}: {count:>4} ({pct:>5.1f}%)")

    # 2. Latency Breakdown
    total_latencies = [
        r["total_latency_ms"] for r in records if r.get("total_latency_ms") is not None
    ]

    if total_latencies:
        avg_total = sum(total_latencies) / len(total_latencies)
        p50_total = percentiles(total_latencies, 50)
        p95_total = percentiles(total_latencies, 95)
        max_total = max(total_latencies)

        print("\n--- LATENCY (TOTAL END-TO-END) ---")
        print(f"  Average           : {avg_total / 1000:6.2f} s ({avg_total:7.1f} ms)")
        print(f"  P50 (Median)      : {p50_total / 1000:6.2f} s ({p50_total:7.1f} ms)")
        print(f"  P95               : {p95_total / 1000:6.2f} s ({p95_total:7.1f} ms)")
        print(f"  Max               : {max_total / 1000:6.2f} s ({max_total:7.1f} ms)")

    # Latency by Execution Path
    print("\n--- LATENCY BY EXECUTION PATH (AVERAGE) ---")
    for path in sorted(paths.keys()):
        path_latencies = [
            r["total_latency_ms"]
            for r in records
            if r.get("execution_path") == path and r.get("total_latency_ms") is not None
        ]
        if path_latencies:
            avg_p = sum(path_latencies) / len(path_latencies)
            print(f"  {path:<20}: {avg_p / 1000:6.2f} s ({avg_p:7.1f} ms)")

    # 3. Pipeline Stages
    intent_lat = [
        r["intent_latency_ms"]
        for r in records
        if r.get("intent_latency_ms") is not None
    ]
    emb_lat = [
        r["embedding_latency_ms"]
        for r in records
        if r.get("embedding_latency_ms") is not None
    ]
    qdrant_lat = [
        r["qdrant_latency_ms"]
        for r in records
        if r.get("qdrant_latency_ms") is not None
    ]
    llm_lat = [
        r["llm_latency_ms"] for r in records if r.get("llm_latency_ms") is not None
    ]

    print("\n--- PIPELINE STAGE LATENCIES (AVERAGE) ---")
    if intent_lat:
        avg_intent = sum(intent_lat) / len(intent_lat)
        print(f"  Intent Classifier : {avg_intent:7.2f} ms")
    if emb_lat:
        avg_emb = sum(emb_lat) / len(emb_lat)
        print(f"  Embedding Compute : {avg_emb:7.2f} ms")
    if qdrant_lat:
        avg_qdrant = sum(qdrant_lat) / len(qdrant_lat)
        print(
            f"  Qdrant Search     : {avg_qdrant / 1000:6.2f} s ({avg_qdrant:7.1f} ms)"
        )
    if llm_lat:
        avg_llm = sum(llm_lat) / len(llm_lat)
        print(f"  LLM Generation    : {avg_llm / 1000:6.2f} s ({avg_llm:7.1f} ms)")

    # 4. Retrieval & Similarity Metrics
    top_scores = [
        r["top_similarity_score"]
        for r in records
        if r.get("top_similarity_score") is not None
    ]
    chunks_retrieved = [
        r["chunks_after_threshold"]
        for r in records
        if r.get("chunks_after_threshold") is not None
    ]
    doc_filter_applied = [
        r for r in records if r.get("document_filter_applied") is True
    ]

    print("\n--- RETRIEVAL & VECTORSTORE METRICS ---")
    if top_scores:
        avg_top_score = sum(top_scores) / len(top_scores)
        print(f"  Average Top Cosine Similarity : {avg_top_score:.4f}")
    if chunks_retrieved:
        avg_chunks = sum(chunks_retrieved) / len(chunks_retrieved)
        print(f"  Avg Chunks Selected/Query    : {avg_chunks:.2f}")

    legal_reqs = sum(1 for r in records if r.get("intent") == "LEGAL")
    if legal_reqs > 0:
        filter_rate = (len(doc_filter_applied) / legal_reqs) * 100
        print(f"  Doc Filter Application Rate  : {filter_rate:.1f}%")

    # 5. Outcome & Reliability
    answers_gen = sum(1 for r in records if r.get("answer_generated") is True)
    fallbacks = sum(1 for r in records if r.get("fallback_triggered") is True)
    ambiguities = sum(1 for r in records if r.get("ambiguity_triggered") is True)
    errors = sum(1 for r in records if r.get("error") is True)

    print("\n--- OUTCOMES & SAFETY METRICS ---")
    print(
        f"  Answer Generation Rate       : {(answers_gen / total_records) * 100:.1f}%"
    )
    print(f"  Fallback Rejection Rate      : {(fallbacks / total_records) * 100:.1f}%")
    print(
        f"  Ambiguity Handling Rate      : {(ambiguities / total_records) * 100:.1f}%"
    )
    print(f"  Error Rate                   : {(errors / total_records) * 100:.1f}%")
    print("=====================================\n")


if __name__ == "__main__":
    log_path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    analyze_performance_logs(log_path_arg)
