from modules.ingestion.pdf_parser import extract_text_from_pdf
from modules.embedding.embedder import Embedder
from modules.vector_store.faiss_indexer import FaissVectorStore
from modules.ingestion.text_chunker import chunk_text
  # You can create this if needed

def ingest_pdf_to_faiss(pdf_path):
    # 1. Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    print("extracted------------------------")
    if not text.strip():
        print("No text extracted from PDF.")
        return

    # 2. Split text into chunks (e.g., ~500 tokens per chunk)
    chunks = chunk_text(text)
    print("chunked------------------------")

    # 3. Generate embeddings for all chunks at once (batch)
    embedder = Embedder()
    embeddings = embedder.embed(chunks)  # embeddings is a list of vectors
    print("embeddings created------------------------")

    # 4. Load or create Faiss store and add chunks + embeddings
    store = FaissVectorStore(dim=384)  # adjust dim if using other embedding model
    store.load()                       # load existing index if any
    print("loaded------------------------")

    # 5. Add all embeddings and corresponding chunks (metadata) in one call
    store.add(embeddings, chunks)
    print("embeddings added------------------------")
    store.save()

    print(f"Successfully ingested {len(chunks)} chunks from {pdf_path}")

# Example usage:
if __name__ == "__main__":
    pdf_file_path = r"samples/140309_3_Instructions to students for End Term Examination May 2025-Online Mode.pdf"
    ingest_pdf_to_faiss(pdf_file_path)
