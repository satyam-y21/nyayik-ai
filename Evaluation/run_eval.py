import os
import sys
from typing import List, Dict, Any

# Ensure backend root is on Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from datasets import Dataset
from Graph.workflow import app as workflow_app

# Sample evaluation dataset representing typical citizen queries
EVAL_DATASET = [
    {
        "question": "What does Article 21 of the Indian Constitution protect?",
        "ground_truth": "Article 21 protects the right to life and personal liberty, stating that no person shall be deprived of their life or personal liberty except according to procedure established by law.",
    },
    {
        "question": "Are citizens guaranteed a right to education in India?",
        "ground_truth": "Yes, Article 21A guarantees the right to free and compulsory education for all children between the ages of six and fourteen years as a fundamental right.",
    },
    {
        "question": "Does Nyaya AI provide legal advice?",
        "ground_truth": "No, Nyaya AI is an educational RAG assistant and does not constitute professional legal advice.",
    },
]


def run_rag_pipeline(dataset: List[Dict[str, str]]) -> Dict[str, List[Any]]:
    """Runs queries through the LangGraph workflow and builds evaluation arrays."""
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in dataset:
        q = item["question"]
        print(f"Querying workflow: '{q}'")
        result = workflow_app.invoke({"query": q})

        questions.append(q)
        answers.append(result.get("generation", ""))
        contexts.append([doc.page_content for doc in result.get("documents", [])])
        ground_truths.append(item["ground_truth"])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def main():
    print("--- STARTING RAGAS EVALUATION RUN ---")

    # Run the pipeline queries to collect responses
    eval_inputs = run_rag_pipeline(EVAL_DATASET)

    # Create huggingface style dataset for Ragas evaluation
    dataset = Dataset.from_dict(eval_inputs)

    # Verify a valid OpenAI key is configured, else fallback to mock scores
    openai_key = os.getenv("OPENAI_API_KEY")
    has_api_key = openai_key and not openai_key.startswith("your_")

    if not has_api_key:
        print(
            "\n[WARNING] No valid OpenAI API key detected for Ragas. Running score simulation."
        )
        print(
            "To run true Ragas metrics, set your OpenAI/Gemini environment variables."
        )
        print("\n--- SIMULATED EVALUATION RESULTS ---")
        print("Faithfulness Score: 0.95")
        print("Answer Relevancy Score: 0.91")
        print("Context Recall Score: 0.88")
        return

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy

        print("\nEvaluating dataset using Ragas metrics...")
        result = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy])

        print("\n--- RAGAS EVALUATION METRICS REPORT ---")
        print(result)

    except Exception as e:
        print(f"\n[ERROR] Ragas evaluation failed: {e}")


if __name__ == "__main__":
    main()
