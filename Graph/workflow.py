import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional  # noqa: UP035

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from Backend.app.config import settings
from Backend.app.core.metrics import IST, get_metrics_logger
from Backend.app.utils.lifecycle import LifecycleManager
from Backend.app.utils.models import get_llm
from Prompts.prompt_templates import (
    GENERAL_LEGAL_PROMPT,
    GROUNDED_SYSTEM_PROMPT,
    OUT_OF_DOMAIN_PROMPT,
)
from Vectorstore.retriever import (
    detect_document_filter,
    get_retriever,
    normalize_scenario_query,
)


class GraphState(TypedDict, total=False):
    query: str
    documents: List[Document]
    generation: str
    intent: str
    response_mode: str  # GROUNDED | GENERAL_INFORMATION | CONVERSATIONAL | OUT_OF_DOMAIN | AMBIGUITY | ERROR
    history: Optional[List[Dict[str, Any]]]
    _metrics: Dict[str, Any]


def format_chat_history(
    history: Optional[List[Dict[str, Any]]], max_turns: int = 20
) -> str:
    """Turn recent history into a string the LLM can read."""
    if not history:
        return "None"

    formatted_turns = []
    recent = history[-max_turns:]
    for msg in recent:
        role = str(msg.get("role", "user")).capitalize()
        content = str(msg.get("content", "")).strip()
        if content:
            if len(content) > 300:
                content = content[:300] + "..."
            formatted_turns.append(f"{role}: {content}")

    return "\n".join(formatted_turns) if formatted_turns else "None"


FAST_CONVERSATIONAL_WORDS = {
    "hi",
    "hello",
    "hey",
    "namaste",
    "greetings",
    "thanks",
    "thankyou",
    "thank you",
}
FAST_CONVERSATIONAL_REGEX = re.compile(
    r"^(hi|hello|hey|namaste|greetings|good\s*(morning|afternoon|evening|night)|"
    r"who\s+are\s+you|what\s+(can|do)\s+you\s+do(\s+for\s+me)?|"
    r"what\s+are\s+you|tell\s+me\s+about\s+yourself|how\s+can\s+you\s+help)[\s?!.]*$",
    re.IGNORECASE,
)

FAST_LEGAL_REGEX = re.compile(
    r"\b(article|section|sec|art|act|law|laws|right|rights|police|arrest|arrested|jail|prison|"
    r"court|courts|fir|bail|offence|offense|crime|crimes|punishment|constitution|constitutional|"
    r"bns|bnss|bsa|ipc|crpc|cpc|evidence|judge|judges|advocate|lawyer|legal|legally|stolen|theft|"
    r"murder|assault|assaulted|injury|injuries|fine|fines|penalty|magistrate|warrant|complaint|"
    r"petition|high court|supreme court|property|tenant|landlord|consumer|divorce|dowry|bribe|"
    r"bribery|cheating|fraud|harassment|custody|searched|seize|seized|witness|accused|victim|"
    r"interrogation|questioning|beaten|slapped|handcuff|handcuffed|suicide|extortion|defamation|"
    r"copyright|patent|trademark|tax|evasion|affidavit|notary|tribunal|jurisdiction|verdict)\b",
    re.IGNORECASE,
)

FAST_OUT_OF_DOMAIN_REGEX = re.compile(
    r"\b(capital of|population of|weather in|distance between|recipe for|how to (bake|cook|make a cake|swim|fly)|"
    r"who (won|directed|starred|sang)|lyrics of|translate|synonym of|antonym of|"
    r"write a (python|javascript|code|script|program|essay|poem)|code for|factorial of|"
    r"what is the distance|how far is|who is the (president|prime minister|ceo|king|queen) of)\b",
    re.IGNORECASE,
)


def classify_intent_node(state: GraphState) -> Dict[str, Any]:
    """Classify query as legal, conversational, or out_of_domain."""
    print("classifying intent...")
    query = state["query"].strip()
    t0 = time.perf_counter()

    # fast-path: greetings & conversational
    if query.lower() in FAST_CONVERSATIONAL_WORDS or FAST_CONVERSATIONAL_REGEX.search(
        query
    ):
        intent = "conversational"
        t_intent_ms = (time.perf_counter() - t0) * 1000
        print(f"Fast-path detected intent: {intent.upper()} ({t_intent_ms:.2f}ms)")
        metrics = state.get("_metrics", {})
        metrics["intent"] = intent.upper()
        metrics["intent_latency_ms"] = round(t_intent_ms, 2)
        return {"intent": intent, "query": query, "_metrics": metrics}

    # fast-path: clearly non-legal queries
    if FAST_OUT_OF_DOMAIN_REGEX.search(query):
        intent = "out_of_domain"
        t_intent_ms = (time.perf_counter() - t0) * 1000
        print(f"Fast-path detected intent: {intent.upper()} ({t_intent_ms:.2f}ms)")
        metrics = state.get("_metrics", {})
        metrics["intent"] = intent.upper()
        metrics["intent_latency_ms"] = round(t_intent_ms, 2)
        return {"intent": intent, "query": query, "_metrics": metrics}

    # fast-path: obvious legal terms present
    if FAST_LEGAL_REGEX.search(query):
        intent = "legal"
        t_intent_ms = (time.perf_counter() - t0) * 1000
        print(f"Fast-path detected intent: {intent.upper()} ({t_intent_ms:.2f}ms)")
        metrics = state.get("_metrics", {})
        metrics["intent"] = intent.upper()
        metrics["intent_latency_ms"] = round(t_intent_ms, 2)
        return {"intent": intent, "query": query, "_metrics": metrics}

    # fallback: LLM classification (3s timeout)
    intent_prompt = (
        "You are an intent classifier for Nyayik AI, an Indian Legal Assistant.\n"
        "Classify the user query into EXACTLY ONE of three categories:\n"
        "1. 'conversational': Greetings, pleasantries, small talk (e.g., 'hi', 'hello', 'who are you', 'what can you do').\n"
        "2. 'out_of_domain': General non-legal queries completely unrelated to law, rights, acts, or procedures (e.g., 'capital of France', 'how to bake a cake', 'who won the cricket match').\n"
        "3. 'legal': Any legal question, rights query, statutory act, police scenario, court procedure, or situation-based query (e.g., 'Article 21', 'police slapped me', 'WhatsApp evidence', 'Section 378').\n\n"
        "Output ONLY the category name ('conversational', 'out_of_domain', or 'legal') with no extra text or explanation.\n\n"
        f"Query: {query}\n"
        "Category:"
    )

    try:
        llm = get_llm()
        response = llm.invoke(intent_prompt, config={"timeout": 3.0})
        raw_intent = response.content.strip().lower()

        if "conversational" in raw_intent:
            intent = "conversational"
        elif "out_of_domain" in raw_intent:
            intent = "out_of_domain"
        else:
            intent = "legal"
    except Exception as e:
        print(f"Intent classification failed ({e}), defaulting to legal")
        intent = "legal"

    t_intent_ms = (time.perf_counter() - t0) * 1000
    print(f"Detected intent: {intent.upper()} ({t_intent_ms:.2f}ms)")

    metrics = state.get("_metrics", {})
    metrics["intent"] = intent.upper()
    metrics["intent_latency_ms"] = round(t_intent_ms, 2)

    return {"intent": intent, "query": query, "_metrics": metrics}


def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """Fetch relevant docs from vector store."""
    print("retrieving documents...")
    query = state["query"]
    search_query = normalize_scenario_query(query)
    metrics = state.get("_metrics", {})
    t0 = time.perf_counter()

    doc_filter = detect_document_filter(search_query)

    try:
        embeddings = LifecycleManager.get_embeddings()
        t_emb_0 = time.perf_counter()
        _ = embeddings.embed_query(search_query)
        t_emb_ms = (time.perf_counter() - t_emb_0) * 1000
    except Exception:
        t_emb_ms = 0.0

    retriever = get_retriever()
    threshold = settings.RETRIEVAL_SCORE_THRESHOLD
    documents = retriever.search(search_query, k=4, threshold=threshold)
    t_retrieve_ms = (time.perf_counter() - t0) * 1000
    t_qdrant_ms = max(0.0, t_retrieve_ms - t_emb_ms)

    metrics["embedding_latency_ms"] = round(t_emb_ms, 2)
    metrics["qdrant_latency_ms"] = round(t_qdrant_ms, 2)
    metrics["retrieval_threshold"] = threshold
    metrics["retrieval_attempted"] = True
    metrics["document_filter_applied"] = doc_filter is not None
    metrics["identified_document_filter"] = doc_filter
    metrics["chunks_after_threshold"] = len(documents)

    scores = [
        doc.metadata.get("similarity_score")
        for doc in documents
        if doc.metadata.get("similarity_score") is not None
    ]
    if scores:
        metrics["top_similarity_score"] = round(max(scores), 4)
        metrics["lowest_selected_similarity_score"] = round(min(scores), 4)

    # Check for ambiguity
    section_number_match = re.search(
        r"\b(?:section|article|sec|art)?\s*(\d+[A-Z]*)\b", query, re.IGNORECASE
    )
    ambiguity_flag = False
    ambiguous_msg = ""

    if section_number_match:
        num = section_number_match.group(1).lower()
        matched_sources = set()
        for doc in documents:
            sec_meta = doc.metadata.get("section", "").lower()
            if num in sec_meta:
                matched_sources.add(doc.metadata.get("source_file"))

        if len(matched_sources) > 1:
            ambiguity_flag = True
            docs_readable = [
                s.replace("_", " ").replace(".pdf", "") for s in matched_sources
            ]
            ambiguous_msg = (
                f"I found matches for '{section_number_match.group(0)}' in both the "
                f"{', and the '.join(docs_readable)}. "
                "Could you please specify which document you are referring to?"
            )

    if ambiguity_flag:
        print("Ambiguity detected! Returning clarifying prompt.")
        metrics["execution_path"] = "AMBIGUITY_CLARIFICATION"
        metrics["response_mode"] = "AMBIGUITY"
        metrics["ambiguity_triggered"] = True
        metrics["retrieval_accepted"] = False
        return {
            "documents": [],
            "query": query,
            "generation": ambiguous_msg,
            "response_mode": "AMBIGUITY",
            "_metrics": metrics,
        }

    return {
        "documents": documents,
        "query": query,
        "generation": "",
        "_metrics": metrics,
    }


def generate_grounded_node(state: GraphState) -> Dict[str, Any]:
    """Mode A: answer from retrieved evidence."""
    print("generating grounded answer (mode A)")
    query = state["query"]
    documents = state["documents"]
    metrics = state.get("_metrics", {})
    t0 = time.perf_counter()

    context_text = "\n\n".join(
        [
            f"[Source: {doc.metadata.get('source_file', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}]\n{doc.page_content}"
            for doc in documents
        ]
    )

    chat_history_str = format_chat_history(state.get("history"))
    formatted_prompt = GROUNDED_SYSTEM_PROMPT.format(
        context=context_text, query=query, chat_history=chat_history_str
    )

    try:
        llm = get_llm()
        response = llm.invoke(formatted_prompt)
        generation = response.content
    except Exception as e:
        primary_provider = settings.LLM_PROVIDER.lower()
        if primary_provider != "ollama":
            print(
                f"[WARNING] Primary LLM ({primary_provider}) failed: {e}. Falling back to local Ollama (llama3)..."
            )
            try:
                from langchain_community.chat_models import ChatOllama

                ollama_llm = ChatOllama(model="llama3", temperature=0)
                response = ollama_llm.invoke(formatted_prompt)
                generation = f"{response.content}\n\n*(Note: Response generated via local Ollama fallback)*"
            except Exception as ollama_err:
                print(f"[ERROR] Ollama fallback failed: {ollama_err}")
                raise e
        else:
            raise e

    # check if LLM flagged insufficient corpus evidence
    refusal_phrases = [
        "[INSUFFICIENT_CORPUS_EVIDENCE]",
        "do not contain sufficient information",
        "cannot be established from the current evidence",
        "provided legal documents do not contain",
        "provided context does not contain",
    ]
    if any(phrase in generation.lower() for phrase in refusal_phrases):
        print("Insufficient corpus evidence, rerouting to mode B")
        return generate_general_legal_node(state)

    t_gen_ms = (time.perf_counter() - t0) * 1000
    metrics["llm_latency_ms"] = round(t_gen_ms, 2)
    metrics["execution_path"] = "LEGAL_GROUNDED"
    metrics["response_mode"] = "GROUNDED"
    metrics["retrieval_accepted"] = True
    metrics["answer_generated"] = True

    return {
        "generation": generation,
        "documents": documents,
        "query": query,
        "response_mode": "GROUNDED",
        "_metrics": metrics,
    }


def generate_general_legal_node(state: GraphState) -> Dict[str, Any]:
    """Mode B: general legal info when corpus is insufficient."""
    print("generating general legal info (mode B)")
    query = state["query"]
    metrics = state.get("_metrics", {})
    t0 = time.perf_counter()

    chat_history_str = format_chat_history(state.get("history"))
    formatted_prompt = GENERAL_LEGAL_PROMPT.format(
        query=query, chat_history=chat_history_str
    )

    try:
        llm = get_llm()
        response = llm.invoke(formatted_prompt)
        generation = response.content
    except Exception as e:
        primary_provider = settings.LLM_PROVIDER.lower()
        if primary_provider != "ollama":
            print(f"[WARNING] Primary LLM failed: {e}. Falling back to local Ollama...")
            try:
                from langchain_community.chat_models import ChatOllama

                ollama_llm = ChatOllama(model="llama3", temperature=0)
                response = ollama_llm.invoke(formatted_prompt)
                generation = f"{response.content}\n\n*(Note: Generated via local Ollama fallback)*"
            except Exception as ollama_err:
                print(f"[ERROR] Ollama fallback failed: {ollama_err}")
                raise e
        else:
            raise e

    t_gen_ms = (time.perf_counter() - t0) * 1000
    metrics["llm_latency_ms"] = round(t_gen_ms, 2)
    metrics["execution_path"] = "GENERAL_LEGAL_INFO"
    metrics["response_mode"] = "GENERAL_INFORMATION"
    metrics["retrieval_accepted"] = False
    metrics["fallback_triggered"] = True
    metrics["fallback_reason"] = "INSUFFICIENT_CORPUS_EVIDENCE"
    metrics["answer_generated"] = True

    return {
        "generation": generation,
        "documents": [],
        "query": query,
        "response_mode": "GENERAL_INFORMATION",
        "_metrics": metrics,
    }


def generate_out_of_domain_node(state: GraphState) -> Dict[str, Any]:
    """Politely decline non-legal queries."""
    print("out of domain response")
    query = state["query"]
    metrics = state.get("_metrics", {})
    t0 = time.perf_counter()

    formatted_prompt = OUT_OF_DOMAIN_PROMPT.format(query=query)

    try:
        llm = get_llm()
        response = llm.invoke(formatted_prompt)
        generation = response.content
    except Exception as e:
        generation = "I am Nyayik AI, specialized in Indian legal and constitutional information. Please ask a question related to Indian law, rights, or acts."

    t_gen_ms = (time.perf_counter() - t0) * 1000
    metrics["llm_latency_ms"] = round(t_gen_ms, 2)
    metrics["execution_path"] = "OUT_OF_DOMAIN"
    metrics["response_mode"] = "OUT_OF_DOMAIN"
    metrics["retrieval_attempted"] = False
    metrics["retrieval_accepted"] = False
    metrics["answer_generated"] = True

    return {
        "generation": generation,
        "documents": [],
        "query": query,
        "response_mode": "OUT_OF_DOMAIN",
        "_metrics": metrics,
    }


def generate_chat_node(state: GraphState) -> Dict[str, Any]:
    """Handle greetings and 'what can you do' questions."""
    print("generating chat response")
    query = state["query"].strip()
    metrics = state.get("_metrics", {})
    t0 = time.perf_counter()

    clean_query = query.lower().strip()
    if (
        clean_query in FAST_CONVERSATIONAL_WORDS
        or clean_query
        in {"good morning", "good afternoon", "good evening", "hi there", "hello there"}
        or FAST_CONVERSATIONAL_REGEX.match(clean_query)
    ):
        generation = (
            "Namaste! I am Nyayik AI, your Indian Legal Information Assistant.\n\n"
            "You can ask me about anything related to Indian law — whether it is a real situation you are facing, a question about your rights, or something you simply want to understand better. For example:\n\n"
            "- *A police officer stopped me and searched my phone — is that legal?*\n"
            "- *What are my rights if I am arrested?*\n"
            "- *What does Article 21 of the Constitution say?*\n"
            "- *How do I file a consumer complaint?*\n\n"
            "What would you like to know?"
        )
        t_gen_ms = (time.perf_counter() - t0) * 1000
        metrics["llm_latency_ms"] = round(t_gen_ms, 2)
        metrics["execution_path"] = "CONVERSATIONAL_FASTPATH"
        metrics["response_mode"] = "CONVERSATIONAL"
        metrics["retrieval_attempted"] = False
        metrics["retrieval_accepted"] = False
        metrics["answer_generated"] = True
        return {
            "generation": generation,
            "documents": [],
            "query": query,
            "response_mode": "CONVERSATIONAL",
            "_metrics": metrics,
        }

    chat_prompt = (
        "You are Nyayik AI, an Indian Legal Information Assistant.\n"
        "The user is greeting you or asking what you can help with.\n\n"
        "Respond warmly and conversationally. Your response should:\n"
        "1. Briefly introduce yourself as Nyayik AI, an Indian Legal Information Assistant.\n"
        "2. Tell the user what kinds of questions or situations they can bring to you — focus on the USER'S needs, not on how the system works internally. For example: police encounters, fundamental rights, property disputes, consumer issues, digital evidence, statutory procedures, constitutional questions.\n"
        "3. Give 3–4 example questions a real user might ask, phrased from the user's perspective (not academic).\n"
        "4. End with a warm, simple invitation for the user to share their question or situation.\n\n"
        "IMPORTANT:\n"
        "- Do NOT mention technical terms like 'RAG', 'corpus', 'indexed documents', 'Mode A', 'Mode B', or 'dual-source architecture'.\n"
        "- Do NOT include any educational disclaimer — the application already has one displayed.\n"
        "- Keep the response concise and natural. No long lists or headers.\n\n"
        f"User Query: {query}"
    )

    try:
        llm = get_llm()
        response = llm.invoke(chat_prompt)
        generation = response.content
    except Exception:  # noqa: BLE001
        generation = (
            "Namaste! I am Nyayik AI, your Indian Legal Information Assistant.\n\n"
            "You can ask me about anything related to Indian law — whether it is a real situation you are facing, a question about your rights, or something you simply want to understand better. For example:\n\n"
            "- *A police officer stopped me and searched my phone — is that legal?*\n"
            "- *What are my rights if I am arrested?*\n"
            "- *What does Article 21 of the Constitution say?*\n"
            "- *How do I file a consumer complaint?*\n\n"
            "What would you like to know?"
        )

    t_gen_ms = (time.perf_counter() - t0) * 1000
    metrics["llm_latency_ms"] = round(t_gen_ms, 2)
    metrics["execution_path"] = "CONVERSATIONAL"
    metrics["response_mode"] = "CONVERSATIONAL"
    metrics["retrieval_attempted"] = False
    metrics["retrieval_accepted"] = False
    metrics["answer_generated"] = True

    return {
        "generation": generation,
        "documents": [],
        "query": query,
        "response_mode": "CONVERSATIONAL",
        "_metrics": metrics,
    }


def route_by_intent(state: GraphState) -> str:
    """Pick next node based on intent."""
    intent = state.get("intent", "legal")
    if intent == "conversational":
        return "chat"
    elif intent == "out_of_domain":
        return "out_of_domain"
    return "retrieve"


def decide_to_generate(state: GraphState) -> str:
    """Route to grounded or general based on retrieval results."""
    print("deciding pathway...")
    if state.get("generation"):
        print("bypass (direct response set)")
        return "bypass"

    documents = state.get("documents", [])
    if not documents:
        print("-> general legal (mode B)")
        return "generate_general_legal"
    else:
        print("-> grounded (mode A)")
        return "generate_grounded"


workflow = StateGraph(GraphState)


workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate_grounded", generate_grounded_node)
workflow.add_node("generate_general_legal", generate_general_legal_node)
workflow.add_node("generate_out_of_domain", generate_out_of_domain_node)
workflow.add_node("generate_chat", generate_chat_node)


workflow.set_entry_point("classify_intent")


workflow.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "chat": "generate_chat",
        "out_of_domain": "generate_out_of_domain",
        "retrieve": "retrieve",
    },
)


workflow.add_conditional_edges(
    "retrieve",
    decide_to_generate,
    {
        "generate_grounded": "generate_grounded",
        "generate_general_legal": "generate_general_legal",
        "bypass": END,
    },
)


workflow.add_edge("generate_grounded", END)
workflow.add_edge("generate_general_legal", END)
workflow.add_edge("generate_out_of_domain", END)
workflow.add_edge("generate_chat", END)


compiled_workflow = workflow.compile()


class InstrumentedWorkflow:
    """Wraps the compiled graph to log metrics per request."""

    def invoke(self, inputs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        start_total = time.perf_counter()
        query_str = inputs.get("query", "")

        metrics_record = {
            "timestamp": datetime.now(IST).isoformat(),
            "request_id": str(uuid.uuid4()),
            "query_text": query_str,
            "query_length": len(query_str),
            "execution_path": "UNKNOWN",
            "response_mode": "UNKNOWN",
            "retrieval_attempted": False,
            "retrieval_accepted": False,
            "error": False,
        }

        inputs["_metrics"] = metrics_record

        try:
            result = compiled_workflow.invoke(inputs, config=config)
            total_ms = (time.perf_counter() - start_total) * 1000

            result_metrics = result.get("_metrics", metrics_record)
            result_metrics["total_latency_ms"] = round(total_ms, 2)
            result_metrics["user_query"] = query_str
            result_metrics["response_text"] = result.get("generation", "")
            result_metrics["response_mode"] = result.get(
                "response_mode", result_metrics.get("response_mode", "UNKNOWN")
            )

            get_metrics_logger().log_request(result_metrics)
            return result
        except Exception as e:
            total_ms = (time.perf_counter() - start_total) * 1000
            metrics_record["total_latency_ms"] = round(total_ms, 2)
            metrics_record["user_query"] = query_str
            metrics_record["execution_path"] = "ERROR"
            metrics_record["response_mode"] = "ERROR"
            metrics_record["error"] = True
            metrics_record["error_type"] = type(e).__name__
            get_metrics_logger().log_request(metrics_record)
            raise e


app = InstrumentedWorkflow()
