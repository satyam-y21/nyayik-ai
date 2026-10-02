import os
import re
from typing import Any, Dict, List, Optional  # noqa: UP035

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from Backend.app.config import settings
from Graph.workflow import app as workflow_app

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    history: Optional[List[Dict[str, Any]]] = None


class SourceDetail(BaseModel):
    file: str
    section: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDetail]
    response_mode: str
    disclaimer: Optional[str] = None


PRONOUN_FOLLOWUP_INDICATORS = {
    "this",
    "that",
    "it",
    "they",
    "them",
    "these",
    "those",
    "same",
    "such",
    "above",
    "mentioned",
    "he",
    "she",
    "his",
    "her",
    "their",
    "again",
    "explain",
    "elaborate",
    "further",
    "more",
    "details",
    "what about",
    "can they",
    "is it",
    "does it",
    "why",
    "how come",
    "and",
}


def build_effective_query(query: str, history: Optional[List[Dict[str, Any]]]) -> str:
    """Expand short follow-up queries with session context for better retrieval."""
    if not history:
        return query

    clean_q = query.strip().lower()
    words = set(re.findall(r"\b\w+\b", clean_q))

    # Check if query is self-contained (contains explicit section/article/act references)
    is_self_contained_legal = bool(
        re.search(
            r"\b(?:section|article|sec|art|bns|bnss|bsa|constitution|act|ipc|crpc)\s*\d*",
            clean_q,
            re.IGNORECASE,
        )
    )

    # Only expand if it is not a self-contained legal query AND (has follow-up indicator OR is very short non-legal phrase)
    has_indicator = bool(words.intersection(PRONOUN_FOLLOWUP_INDICATORS))
    is_short_continuation = len(words) <= 5 and not is_self_contained_legal

    if not is_self_contained_legal and (has_indicator or is_short_continuation):
        user_queries = []
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "").strip()
            if role == "user" and content and content.lower() != clean_q:
                base_q = content.split(" — ")[-1] if " — " in content else content
                if base_q.lower() not in {
                    "hi",
                    "hello",
                    "hey",
                    "explain further",
                    "tell me more",
                    "elaborate",
                }:
                    user_queries.append(base_q)

        if user_queries:
            primary_topic = user_queries[0]
            latest_topic = user_queries[-1]

            if primary_topic.lower() == latest_topic.lower():
                expanded_context = primary_topic
            else:
                expanded_context = f"{primary_topic} ({latest_topic})"

            print(f"expanded: '{query}' -> '{expanded_context}'")
            return f"{expanded_context} — {query}"

    return query


@router.post("/query", response_model=QueryResponse)
async def query_assistant(payload: QueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        effective_query = build_effective_query(payload.query, payload.history)
        result = workflow_app.invoke(
            {"query": effective_query, "history": payload.history}
        )
        response_mode = result.get("response_mode", "GROUNDED")

        sources = []
        if response_mode == "GROUNDED":
            answer_text = result.get("generation", "")
            documents = result.get("documents", [])
            answer_lower = answer_text.lower()

            cited_pages = set()
            for m in re.finditer(r"page[s]?[:\s]+(\d+(?:\s*,\s*\d+)*)", answer_lower):
                for p in m.group(1).split(","):
                    p_str = p.strip()
                    if p_str.isdigit():
                        cited_pages.add(int(p_str))

            seen_keys = set()
            for doc in documents:
                sec = str(doc.metadata.get("section", "")).lower()
                page_num = doc.metadata.get("page")
                file_name = doc.metadata.get("source_file", "Official Document")

                is_page_cited = page_num is not None and int(page_num) in cited_pages

                is_section_cited = False
                if sec and sec != "general section":
                    sec_num_match = re.search(r"\d+[a-z]*", sec)
                    if sec_num_match:
                        num_str = sec_num_match.group(0)
                        if re.search(r"\b" + re.escape(num_str) + r"\b", answer_lower):
                            is_section_cited = True

                if cited_pages:
                    should_include = is_page_cited
                else:
                    should_include = is_section_cited

                if should_include:
                    sec_display = (
                        f"{str(doc.metadata.get('section', '')).title()} (Page {page_num})"
                        if sec and sec != "general section"
                        else f"Page {page_num}"
                    )
                    dedup_key = (file_name, sec_display)
                    if dedup_key not in seen_keys:
                        seen_keys.add(dedup_key)
                        sources.append(
                            SourceDetail(
                                file=file_name,
                                section=sec_display,
                                text=doc.page_content,
                            )
                        )

            if not sources and documents:
                top_doc = documents[0]
                sec = top_doc.metadata.get("section")
                page_num = top_doc.metadata.get("page", "Unknown")
                file_name = top_doc.metadata.get("source_file", "Official Document")
                sec_display = (
                    f"{str(sec).title()} (Page {page_num})"
                    if sec and sec != "general section"
                    else f"Page {page_num}"
                )
                sources.append(
                    SourceDetail(
                        file=file_name, section=sec_display, text=top_doc.page_content
                    )
                )

        disclaimer = None
        if response_mode == "GENERAL_INFORMATION":
            disclaimer = "General legal information — not verified against Nyayik AI's indexed legal sources."

        raw_answer = result.get("generation", "")
        clean_answer = re.sub(
            r"\s*\[(?:Source:\s*)?[^\]]*?\b(?:page[s]?|pdf)\b[^\]]*?\]",
            "",
            raw_answer,
            flags=re.IGNORECASE,
        )
        clean_answer = re.sub(r"\s*\.\s*\.", ".", clean_answer)
        clean_answer = re.sub(r"\s+\.", ".", clean_answer).strip()

        if response_mode in ("GROUNDED", "GENERAL_INFORMATION"):
            clean_answer = re.sub(
                r"^(?:namaste|hello|greetings|hi)[!\s,]*(?:i am nyayik ai[^.\n]*[.!]?)?\s*",
                "",
                clean_answer,
                flags=re.IGNORECASE,
            ).strip()
            clean_answer = re.sub(
                r"^(?:based on|according to)\s+(?:the\s+)?(?:official\s+|indexed\s+|retrieved\s+|provided\s+)?(?:legal\s+)?(?:document[s]?|context|evidence|database|record[s]?)[^.\n]*[.,:]?\s*",
                "",
                clean_answer,
                flags=re.IGNORECASE,
            ).strip()
            if clean_answer and clean_answer[0].islower():
                clean_answer = clean_answer[0].upper() + clean_answer[1:]

        return QueryResponse(
            answer=clean_answer,
            sources=sources,
            response_mode=response_mode,
            disclaimer=disclaimer,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents():
    raw_dir = settings.DATA_RAW_DIR
    if not os.path.exists(raw_dir):
        return {"documents": []}

    # Dynamically scan the directory
    pdf_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".pdf")]
    return {"documents": pdf_files}
