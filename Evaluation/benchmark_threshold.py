import os
import sys
from typing import List, Dict, Any

# Ensure backend root is on Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from Vectorstore.retriever import get_retriever
from Backend.app.config import settings

# Comprehensive legal benchmarking dataset
BENCHMARK_DATASET = {
    "direct_legal": [
        {
            "query": "What are my fundamental rights under Article 21?",
            "expected_section": "Article 21",
        },
        {
            "query": "Is there a right to primary education for children?",
            "expected_section": "Article 21A",
        },
        {
            "query": "Explain Article 14 of the Constitution",
            "expected_section": "Article 14",
        },
        {
            "query": "What does Section 378 of the BNS define?",
            "expected_section": "Section 378",
        },
        {
            "query": "What is defined under Section 319 of Bharatiya Nagarik Suraksha Sanhita?",
            "expected_section": "Section 319",
        },
        {
            "query": "How is oral evidence defined in Bharatiya Sakshya Adhiniyam?",
            "expected_section": "Section 59",  # (Or another BSA section from indexing)
        },
    ],
    "conversational_legal": [
        {
            "query": "Hey there! Hope you are doing good. Can you tell me if I have a right to clean environment under Article 21?",
            "expected_section": "Article 21",
        },
        {
            "query": "Please explain to me simply: how does Article 14 make everyone equal?",
            "expected_section": "Article 14",
        },
    ],
    "out_of_scope": [
        "What is the capital of France?",
        "How do I bake a chocolate cake?",
        "Tell me about the history of the internet.",
        "Who wrote the play Hamlet?",
        "What is the speed of light?",
    ],
}


def evaluate_threshold(retriever, threshold: float) -> Dict[str, float]:
    """Runs evaluation queries for a given threshold and returns metrics."""
    # Direct legal metrics
    hits = 0
    rr_sum = 0
    total_direct = len(BENCHMARK_DATASET["direct_legal"])

    for item in BENCHMARK_DATASET["direct_legal"]:
        query = item["query"]
        expected = item["expected_section"].lower()

        # Search
        results = retriever.search(query, k=4, threshold=threshold)

        # Check rank
        hit_rank = 0
        for idx, doc in enumerate(results):
            section = doc.metadata.get("section", "").lower()
            chapter = doc.metadata.get("chapter", "").lower()
            page_content = doc.page_content.lower()

            if expected in section or expected in page_content:
                hit_rank = idx + 1
                break

        if hit_rank > 0:
            hits += 1
            rr_sum += 1 / hit_rank

    hit_rate = hits / total_direct if total_direct > 0 else 0
    mrr = rr_sum / total_direct if total_direct > 0 else 0

    # Conversational legal metrics
    conv_hits = 0
    total_conv = len(BENCHMARK_DATASET["conversational_legal"])
    for item in BENCHMARK_DATASET["conversational_legal"]:
        query = item["query"]
        expected = item["expected_section"].lower()
        results = retriever.search(query, k=4, threshold=threshold)

        for doc in results:
            section = doc.metadata.get("section", "").lower()
            page_content = doc.page_content.lower()
            if expected in section or expected in page_content:
                conv_hits += 1
                break
    conv_hit_rate = conv_hits / total_conv if total_conv > 0 else 0

    # Out of scope rejection metrics
    rejected = 0
    total_oos = len(BENCHMARK_DATASET["out_of_scope"])

    for query in BENCHMARK_DATASET["out_of_scope"]:
        results = retriever.search(query, k=4, threshold=threshold)
        if len(results) == 0:
            rejected += 1

    rejection_rate = rejected / total_oos if total_oos > 0 else 0

    return {
        "hit_rate": hit_rate,
        "mrr": mrr,
        "conv_hit_rate": conv_hit_rate,
        "rejection_rate": rejection_rate,
    }


def main():
    print(f"--- STARTING RETRIEVING BENCHMARK RUN ---")
    provider = settings.VECTOR_DB_PROVIDER.lower()
    print(f"Active Provider: {provider.upper()}")

    try:
        retriever = get_retriever()
    except Exception as e:
        print(f"Failed to load retriever: {e}")
        return

    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.82, 0.85, 0.90]

    print("\n" + "=" * 80)
    print(
        f"{'Threshold':<12} | {'Legal Hit Rate':<15} | {'MRR':<10} | {'Conv Hit Rate':<15} | {'OOS Rejection':<15}"
    )
    print("=" * 80)

    results_summary = {}

    for t in thresholds:
        metrics = evaluate_threshold(retriever, t)
        results_summary[t] = metrics
        print(
            f"{t:<12.2f} | {metrics['hit_rate']:<15.2%} | {metrics['mrr']:<10.4f} | {metrics['conv_hit_rate']:<15.2%} | {metrics['rejection_rate']:<15.2%}"
        )

    print("=" * 80)

    print("\n--- DETAILED ARCHITECTURAL ANNALYSIS ---")
    # Let's find the best threshold: highest legal hit rate while keeping 100% OOS rejection
    best_t = 0.82
    best_score = -1

    for t, m in results_summary.items():
        # We weigh Rejection extremely heavily (must be >= 80% to be considered safe)
        if m["rejection_rate"] >= 0.80:
            score = m["hit_rate"] + m["mrr"]
            if score > best_score:
                best_score = score
                best_t = t

    print(f"Empirically optimal threshold for this dataset is: {best_t:.2f}")
    print(f"At {best_t:.2f}:")
    print(
        f" - Out-Of-Scope Rejection Rate: {results_summary[best_t]['rejection_rate']:.1%}"
    )
    print(f" - Legal Query Hit Rate: {results_summary[best_t]['hit_rate']:.1%}")
    print(f" - Mean Reciprocal Rank (MRR): {results_summary[best_t]['mrr']:.4f}")


if __name__ == "__main__":
    main()
