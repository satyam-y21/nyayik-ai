import os
import sys

# Ensure backend root is on Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Backend.app.config import settings
from Backend.app.utils.document_processor import DocumentProcessor
from Vectorstore.retriever import get_retriever


def main():
    raw_dir = settings.DATA_RAW_DIR
    print(f"Scanning for PDF documents in: {raw_dir}")

    if not os.path.exists(raw_dir):
        print(f"Directory {raw_dir} does not exist. Creating it...")
        os.makedirs(raw_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(
            "No PDF files found in raw data directory. "
            "Please add some PDFs to ingest."
        )
        return

    print(f"Found {len(pdf_files)} PDF file(s) to process.")

    processor = DocumentProcessor()
    all_chunks = []

    for pdf in pdf_files:
        pdf_path = os.path.join(raw_dir, pdf)
        print(f"Processing: {pdf}...")
        try:
            chunks = processor.process_document(pdf_path)
            all_chunks.extend(chunks)
            print(f"Successfully processed {pdf} ({len(chunks)} chunks).")
        except Exception as e:
            print(f"Error processing {pdf}: {e}")

    if not all_chunks:
        print("No chunks generated. Ingestion aborted.")
        return

    provider = settings.VECTOR_DB_PROVIDER.lower()
    print(
        f"Generated {len(all_chunks)} total chunks. "
        f"Creating {provider.upper()} index..."
    )

    try:
        retriever = get_retriever()
        retriever.create_and_save_index(all_chunks)
        print(f"{provider.upper()} index successfully built and saved/uploaded.")
    except Exception as e:
        print(f"Error saving {provider.upper()} index: {e}")


if __name__ == "__main__":
    main()
