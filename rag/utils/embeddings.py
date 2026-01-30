
from sentence_transformers import SentenceTransformer
import numpy as np
class EmbeddingGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def save_embeddings(self, texts: list[str], embeddings: list[list[float]], output_path: str):

        np.savez_compressed(output_path, texts=texts, embeddings=embeddings)