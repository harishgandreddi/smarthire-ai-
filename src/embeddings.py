from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

MODEL_NAME = "all-MiniLM-L6-v2"

class EmbeddingEngine:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.candidate_data = []

    def embed(self, texts: list[str]) -> np.ndarray:
        """Convert list of texts to embedding vectors."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype("float32")

    def build_index(self, candidates: list[dict]):
        """Build FAISS index from candidate resume texts."""
        self.candidate_data = candidates
        texts = [c["raw_text"] for c in candidates]
        embeddings = self.embed(texts)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        return self.index

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search for top-k matching candidates given a job description."""
        if self.index is None or not self.candidate_data:
            return []

        query_vec = self.embed([query])
        distances, indices = self.index.search(query_vec, min(top_k, len(self.candidate_data)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.candidate_data):
                candidate = self.candidate_data[idx].copy()
                candidate["vector_score"] = float(1 / (1 + dist))
                results.append(candidate)
        return results

    def save(self, path: str = "models/faiss_index"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path + ".index")
        with open(path + ".pkl", "wb") as f:
            pickle.dump(self.candidate_data, f)

    def load(self, path: str = "models/faiss_index"):
        self.index = faiss.read_index(path + ".index")
        with open(path + ".pkl", "rb") as f:
            self.candidate_data = pickle.load(f)
