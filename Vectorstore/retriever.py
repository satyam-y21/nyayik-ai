import re
from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.documents import Document

from Backend.app.config import settings


class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 4, threshold: float = 0.8) -> List[Document]:
        pass


def detect_document_filter(query: str) -> Optional[str]:
    """Map query keywords to a specific source PDF, or None."""
    q = query.lower()

    # Document keyword bindings
    keywords = {
        "indian_constitution.pdf": [
            "constitution",
            "samvidhan",
            "article",
            "fundamental right",
            "fundamental rights",
        ],
        "Bharatiya_Nyaya_Sanhita.pdf": [
            "nyaya sanhita",
            "nyaya",
            "bns",
            "ipc",
            "theft",
            "murder",
            "punishment",
        ],
        "Bharatiya_Nagarik_Suraksha_Sanhita.pdf": [
            "nagarik",
            "suraksha",
            "bnss",
            "crpc",
            "arrest",
            "warrant",
            "bail",
            "investigation",
            "police",
            "police station",
            "custody",
            "handcuff",
        ],
        "Bharatiya_Sakshya_Adhiniyam.pdf": [
            "sakshya",
            "adhiniyam",
            "bsa",
            "evidence",
            "oral evidence",
            "documentary evidence",
            "witness",
        ],
    }

    # Check for keywords using word boundaries
    for doc_name, keys in keywords.items():
        for key in keys:
            if re.search(r"\b" + re.escape(key) + r"\b", q):
                return doc_name

    return None


def normalize_scenario_query(query: str) -> str:
    """Pull legal concepts out of narrative queries for better retrieval."""
    q_lower = query.lower()
    extra_terms = []

    # Police station / arrest / custody scenarios
    if any(
        k in q_lower
        for k in [
            "police station",
            "taken to the police",
            "taken to police",
            "arrested",
            "custody",
            "detained",
            "handcuff",
        ]
    ):
        extra_terms.append(
            "rights of arrested person police station custody detention "
            "grounds of arrest BNSS Section 35 Section 37 Section 47 Article 22"
        )

    # Search / phone / electronic device scenarios
    if any(
        k in q_lower
        for k in [
            "search my phone",
            "searched my phone",
            "check my phone",
            "seize phone",
            "electronic device",
        ]
    ):
        extra_terms.append(
            "police search seizure electronic record phone device "
            "BSA BNSS Section 35 Section 185"
        )

    # Assault / physical violence scenarios
    if any(
        k in q_lower
        for k in ["slapped", "beaten", "attacked", "caused injuries", "assaulted"]
    ):
        extra_terms.append(
            "hurt grievous hurt assault private defence "
            "BNS Section 115 Section 117 Section 38"
        )

    if extra_terms:
        expanded = f"{query} {' '.join(extra_terms)}"
        print(f"scenario expansion: '{query}' -> '{expanded}'")
        return expanded

    return query


def detect_target_section(query: str) -> Optional[str]:
    """Extract 'article X' or 'section X' from query if present."""
    m = re.search(
        r"\b(?:section|article|sec|art)\s*(\d+[A-Z]*)\b", query, re.IGNORECASE
    )
    if m:
        num = m.group(1).lower()
        if re.search(r"\b(?:article|art)\b", query, re.IGNORECASE) or (
            "constitution" in query.lower()
        ):
            return f"article {num}"
        elif re.search(r"\b(?:section|sec)\b", query, re.IGNORECASE):
            return f"section {num}"
        else:
            return f"article {num}"
    return None


def get_retriever() -> BaseRetriever:
    """Return the configured retriever (faiss or qdrant)."""
    provider = settings.VECTOR_DB_PROVIDER.lower()

    if provider == "faiss":
        from Vectorstore.faiss_store import FAISSRetriever

        return FAISSRetriever()
    elif provider == "qdrant":
        from Vectorstore.qdrant_store import QdrantRetriever

        return QdrantRetriever()
    else:
        raise ValueError(f"Unsupported Vector DB Provider: {provider}")
