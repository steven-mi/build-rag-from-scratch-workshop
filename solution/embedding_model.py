import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text: str) -> np.ndarray:
        """Create an embedding for the given text."""
        return self.model.encode(text)


if __name__ == "__main__":
    model = EmbeddingModel()

    def similarity(text1: str, text2: str) -> float:
        """Calculate the cosine similarity between two texts."""
        embedding1 = model.create_embedding(text1)
        embedding2 = model.create_embedding(text2)
        return float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
    
    # Test embeddings and similarity
    print("Embedding Length:", len(model.create_embedding("Things to do in Berlin")))
    print("Similarity (Cappuccino, Coffe with milk):", similarity("Cappuccino", "Coffe with milk"))
    print("Similarity (Cappuccino, Coffe without milk):", similarity("Cappuccino", "Coffe without milk"))
    print("Similarity (Cappuccino, Sparkling Water):", similarity("Cappuccino", "Sparkling Water"))
    print("Similarity (Cappuccino, Italian Hot Drink):", similarity("Cappuccino", "Italian Hot Drink"))
    print("Similarity (Cappuccino, Chinese Hot Drink):", similarity("Cappuccino", "Chinese Hot Drink"))
