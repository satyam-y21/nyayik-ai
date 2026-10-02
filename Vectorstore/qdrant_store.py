from typing import List

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from Backend.app.config import settings
from Backend.app.utils.lifecycle import LifecycleManager
from Vectorstore.retriever import BaseRetriever


class QdrantRetriever(BaseRetriever):
    def __init__(self):
        self.client = LifecycleManager.get_qdrant_client()
        self.embeddings = LifecycleManager.get_embeddings()
        self.collection_name = settings.QDRANT_COLLECTION
        self._db = None

    def _get_db(self) -> QdrantVectorStore:
        """Lazy-init the QdrantVectorStore wrapper."""
        if self._db is None:
            # create collection if missing
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                print(f"Creating Qdrant collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=384,  # matching bge-small dimensions
                        distance=qmodels.Distance.COSINE,
                    ),
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="metadata.source_file",
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="metadata.section",
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )

            self._db = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
        return self._db

    def create_and_save_index(self, documents: List[Document]) -> QdrantVectorStore:
        """Wipe and re-upload all docs to Qdrant."""
        if not documents:
            raise ValueError("No documents provided to index")

        print(f"Recreating collection: {self.collection_name}")
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=384,
                distance=qmodels.Distance.COSINE,
            ),
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="metadata.source_file",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="metadata.section",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )

        db = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        batch_size = 250
        total_docs = len(documents)
        print(f"Uploading {total_docs} vectors to Qdrant (batch size {batch_size})")
        for i in range(0, total_docs, batch_size):
            batch = documents[i : i + batch_size]
            db.add_documents(batch)
            uploaded = min(i + batch_size, total_docs)
            print(f"  {uploaded}/{total_docs} uploaded")

        self._db = db
        return db

    def search(self, query: str, k: int = 4, threshold: float = 0.8) -> List[Document]:
        """Search Qdrant with optional doc filter and section boosting."""
        try:
            db = self._get_db()
            from Vectorstore.retriever import detect_document_filter, detect_target_section

            doc_filter = detect_document_filter(query)
            target_section = detect_target_section(query)

            must_filters = []
            if doc_filter:
                print(f"Filtering by document: {doc_filter}")
                must_filters.append(
                    qmodels.FieldCondition(
                        key="metadata.source_file",
                        match=qmodels.MatchValue(value=doc_filter),
                    )
                )

            filter_cond = (
                qmodels.Filter(must=must_filters) if must_filters else None
            )

            results = db.similarity_search_with_score(
                query, k=k * 2, filter=filter_cond
            )

            # exact section match boost
            exact_matches = []
            if target_section:
                print(f"Targeted section: '{target_section}'")
                sec_must = list(must_filters)
                sec_must.append(
                    qmodels.FieldCondition(
                        key="metadata.section",
                        match=qmodels.MatchValue(value=target_section),
                    )
                )
                try:
                    sec_filter = qmodels.Filter(must=sec_must)
                    exact_res = db.similarity_search_with_score(
                        query, k=k, filter=sec_filter
                    )
                    for doc, score in exact_res:
                        exact_matches.append((doc, float(score)))
                except Exception as ex_err:
                    print(
                        f"[WARNING] Exact payload search for '{target_section}' "
                        f"skipped: {ex_err}"
                    )

            seen_ids = set()
            combined = []
            for doc, score in exact_matches:
                doc_key = (
                    doc.metadata.get("source_file"),
                    doc.metadata.get("page"),
                    doc.metadata.get("section"),
                    doc.page_content[:60],
                )
                if doc_key not in seen_ids:
                    seen_ids.add(doc_key)
                    doc.metadata["similarity_score"] = max(score, 0.95)
                    combined.append(doc)

            for doc, score in results:
                doc_key = (
                    doc.metadata.get("source_file"),
                    doc.metadata.get("page"),
                    doc.metadata.get("section"),
                    doc.page_content[:60],
                )
                if doc_key not in seen_ids:
                    seen_ids.add(doc_key)
                    doc.metadata["similarity_score"] = float(score)
                    combined.append(doc)

            filtered_docs = []
            for doc in combined[:k]:
                score = doc.metadata.get("similarity_score", 0.0)
                sec_label = doc.metadata.get("section_title") or doc.metadata.get(
                    "section", "Unknown"
                )
                print(f"  match: '{sec_label}' score={score:.4f} (min={threshold})")
                if score >= threshold:
                    filtered_docs.append(doc)
            return filtered_docs
        except Exception as e:
            print(f"[ERROR] Qdrant search failed: {e}")
            return []
