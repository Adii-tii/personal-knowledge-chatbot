import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from modules.ingestion.pdf_parser import extract_text_from_pdf
from modules.ingestion.text_chunker import chunk_text
from modules.embedding.embedder import Embedder
from modules.vector_store.faiss_indexer import FaissVectorStore

from docx import Document

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

class FileIngestHandler(FileSystemEventHandler):
    def __init__(self, store, embedder, folder):
        self.store = store
        self.embedder = embedder
        self.folder = folder

    def on_created(self, event):
        if event.is_directory:
            return

        filepath = event.src_path
        ext = os.path.splitext(filepath)[1].lower()

        print(f"[Watcher] Detected new file: {filepath}")

        if ext == ".pdf":
            text = extract_text_from_pdf(filepath)

        # Chunk, embed, and store
        chunks = chunk_text(text)
        vectors = self.embedder.embed(chunks)
        self.store.add(vectors, chunks)
        self.store.save()

        print(f"[Done] Ingested {filepath} and added {len(chunks)} chunks to vector store.")

def start_watching(folder_path):
    embedder = Embedder()
    store = FaissVectorStore(dim=384)
    store.load()

    event_handler = FileIngestHandler(store, embedder, folder_path)
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()

    print(f"🔍 Watching folder: {folder_path}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    folder_path = "samples"
    start_watching(folder_path)
