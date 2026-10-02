import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from Backend.app.config import settings
from Backend.app.utils.lifecycle import LifecycleManager
from Vectorstore.retriever import BaseRetriever


class FAISSRetriever(BaseRetriever):
    def __init__(self, index_dir: str = settings.VECTOR_DB_DIR):
        self.index_dir = index_dir
        self.embeddings = LifecycleManager.get_embeddings()
        self._db = None

    def _get_db(self) -> FAISS:
        """Cache index in memory instead of reloading from disk on every query."""
        if self._db is None:
            if not os.path.exists(os.path.join(self.index_dir, "index.faiss")):
                raise FileNotFoundError(
                    f"FAISS index not found at {self.index_dir}"
                )
            self._db = FAISS.load_local(
                self.index_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        return self._db

    def create_and_save_index(self, documents: List[Document]) -> FAISS:
        """Create a new FAISS index from documents and save it locally."""
        if not documents:
            raise ValueError("No documents provided to index")

        db = FAISS.from_documents(documents, self.embeddings)
        os.makedirs(os.path.dirname(self.index_dir), exist_ok=True)
        db.save_local(self.index_dir)
        self._db = db
        return db

    def search(self, query: str, k: int = 4, threshold: float = 0.6) -> List[Document]:
        """Perform similarity search with relevance score filtering."""
        try:
            db = self._get_db()
            from Vectorstore.retriever import detect_document_filter

            doc_filter = detect_document_filter(query)
            filter_func = None
            if doc_filter:
                print(f"Filtering by document: {doc_filter}")
                filter_func = (
                    lambda metadata: metadata.get("source_file") == doc_filter
                )

            results = db.similarity_search_with_relevance_scores(
                query, k=k, filter=filter_func
            )

            filtered_docs = []
            for doc, score in results:
                sec_label = doc.metadata.get("section", "Unknown")
                print(f"  match: '{sec_label}' score={score:.4f} (min={threshold})")
                if score >= threshold:
                    filtered_docs.append(doc)
            return filtered_docs
        except FileNotFoundError:
            print("FAISS index not found, returning empty results.")
            return []
