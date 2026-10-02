"""
Diagnostic: inspect chunks around Article 21 and page 354 of the Constitution.
Run from project root: python Backend/app/scripts/inspect_chunks.py
"""

import sys, os

sys.path.insert(0, ".")

from Backend.app.utils.document_processor import DocumentProcessor

processor = DocumentProcessor()
print("Processing indian_constitution.pdf...")
chunks = processor.process_document("Data/raw/indian_constitution.pdf")
print(f"Total chunks: {len(chunks)}\n")

# Show all chunks tagged with section containing '21'
print("=" * 80)
print("CHUNKS TAGGED AS 'article 21' (any page):")
print("=" * 80)
for c in chunks:
    if c.metadata.get("section", "").lower() == "article 21":
        print(
            f"  Page: {c.metadata.get('page')} | Chapter: {c.metadata.get('chapter')}"
        )
        print(
            f"  Content (first 200 chars): {c.page_content[:200].replace(chr(10), ' ')}"
        )
        print()

# Show chunks around page 42 (actual article 21 location)
print("=" * 80)
print("CHUNKS ON PAGES 40-45 (around Article 21 content):")
print("=" * 80)
for c in chunks:
    pg = c.metadata.get("page", 0)
    if 40 <= pg <= 45:
        print(
            f"  Page: {pg} | Section: {c.metadata.get('section')} | Chapter: {c.metadata.get('chapter')}"
        )
        print(
            f"  Content (first 300 chars): {c.page_content[:300].replace(chr(10), ' ')}"
        )
        print()

# Show chunks around page 354 (wrong chunk being retrieved)
print("=" * 80)
print("CHUNKS ON PAGES 350-358 (around page 354 - the wrong chunk):")
print("=" * 80)
for c in chunks:
    pg = c.metadata.get("page", 0)
    if 350 <= pg <= 358:
        print(
            f"  Page: {pg} | Section: {c.metadata.get('section')} | Chapter: {c.metadata.get('chapter')}"
        )
        print(
            f"  Content (first 300 chars): {c.page_content[:300].replace(chr(10), ' ')}"
        )
        print()
