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
        # TODO: Retrieve the top_k most similar documents for the given query
        raise NotImplementedError()

    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        # TODO: Implement a function that takes two vectors and compute the cosine similarity
        raise NotImplementedError()

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