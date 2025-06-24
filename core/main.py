from modules.embedding.embedder import Embedder
from modules.vector_store.faiss_indexer import FaissVectorStore
from modules.qa.qa_engine import QAEngine

# Initialize embedder (for encoding queries)
embedder = Embedder()

# Initialize vector store with correct dimension (384 for MiniLM, for example)
store = FaissVectorStore(dim=384)

# Load existing index and metadata
store.load()

# Initialize QA engine with embedder and vector store
qa = QAEngine(embedder, store, "AIzaSyCQOaC4E7HPWq_c3uuJmFZzcaF_FE6O5vU")

# Now just ask questions!
question = input("Type your query: ")
answer = qa.answer_question(question)

print("Question:", question)
print("Answer:", answer)
