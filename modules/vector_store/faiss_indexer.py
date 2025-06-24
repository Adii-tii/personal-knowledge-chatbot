import faiss
import numpy as np
import os
import pickle

class FaissVectorStore:
    def __init__(self, dim, index_path = "modules/vector_store/index.faiss", metadata_path = "modules/vector_store/metadata.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []
    
    def add(self, vectors, metadata_chunks):
        vectors_np  = np.array(vectors).astype("float32")
        self.index.add(vectors_np)
        self.metadata.extend(metadata_chunks)
    
    def search(self, query_vector, top_k = 20):
        query_np = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(query_np, top_k)
        return [(self.metadata[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
    
    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
    
    def load(self):
        if os.path.exists(self.index_path) and os.path.getsize(self.index_path) > 0:
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception as e:
                print(f"[Warning] Failed to load Faiss index: {e}. Creating new index.")
                self.index = faiss.IndexFlatL2(self.dim)
        else:
            print(f"[Info] Index file not found or empty. Creating new index.")
            self.index = faiss.IndexFlatL2(self.dim)

        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"[Warning] Failed to load metadata (empty or corrupted): {e}. Initializing empty metadata.")
                self.metadata = []
        else:
            self.metadata = []

