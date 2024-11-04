import os
import pickle

import numpy as np
import pandas as pd

from typing import List

from embedding_model import EmbeddingModel


class EmbeddingDatabase:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.db = pd.DataFrame([], columns=["text", "text_embedding"])

        state_file = os.getenv("EMBEDDING_DB_HOME")
        if state_file and os.path.exists(state_file):
            self.load_state(state_file)

    def add_documents(self, documents: List[str]):
        """Add documents to the embedding database."""
        data = [
            {
                "text": doc,
                "text_embedding": self.embedding_model.create_embedding(doc),
            }
            for doc in documents
        ]
        df = pd.DataFrame(data)
        self.db = pd.concat([self.db, df], ignore_index=True)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve the top_k most similar documents for the given query."""
        query_embedding = self.embedding_model.create_embedding(query)
        temp = self.db.copy()
        
        temp["query_similarity"] = temp.apply(
            lambda row: self._compute_cosine_similarity(row["text_embedding"], query_embedding),
            axis=1,
        )

        sorted_df = temp.sort_values("query_similarity", ascending=False)
        return list(sorted_df["text"][:top_k])

    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def save_state(self):
        """Save the current state of the database to a file."""
        state_file = os.getenv("EMBEDDING_DB_HOME")
        if state_file:
            with open(state_file, "wb") as f:
                pickle.dump(self.db, f)

    def load_state(self, state_file: str):
        """Load the database state from a file."""
        with open(state_file, "rb") as f:
            self.db = pickle.load(f)


if __name__ == "__main__":
    model = EmbeddingModel()
    db = EmbeddingDatabase(model)
    print(db.retrieve("How do you pick a green?"))