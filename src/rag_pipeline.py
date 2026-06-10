from src.parser import extract_text_from_uploaded, parse_candidate_info
from src.embeddings import EmbeddingEngine
from src.ranker import rank_all_candidates

class SmartHirePipeline:
    def __init__(self):
        self.engine = EmbeddingEngine()
        self.candidates = []

    def load_resumes(self, uploaded_files: list) -> int:
        self.candidates = []
        for f in uploaded_files:
            text = extract_text_from_uploaded(f)
            info = parse_candidate_info(text)
            info["filename"] = f.name
            self.candidates.append(info)
        if self.candidates:
            self.engine.build_index(self.candidates)
        return len(self.candidates)

    def load_resumes_from_text(self, text_inputs: list) -> int:
        self.candidates = []
        for i, text in enumerate(text_inputs):
            if text.strip():
                info = parse_candidate_info(text)
                info["filename"] = f"candidate_{i+1}.txt"
                self.candidates.append(info)
        if self.candidates:
            self.engine.build_index(self.candidates)
        return len(self.candidates)

    def run(self, job_description: str, top_k: int = 10) -> list:
        if not self.candidates:
            return []
        retrieved = self.engine.search(job_description, top_k=top_k)
        ranked = rank_all_candidates(job_description, retrieved)
        return ranked