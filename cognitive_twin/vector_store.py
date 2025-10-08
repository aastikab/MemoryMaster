"""Vector store module using FAISS for similarity search."""

from typing import List, Tuple
import numpy as np
import faiss

class VectorStore:
    def __init__(self, dimension: int = 384):
        """Initialize FAISS index with specified dimension."""
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []  # Store original texts

    def add_embeddings(self, embeddings: np.ndarray, texts: List[str]):
        """Add embeddings and their corresponding texts to the store."""
        if len(embeddings) != len(texts):
            raise ValueError("Number of embeddings must match number of texts")
        
        self.index.add(embeddings)
        self.texts.extend(texts)

    def find_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Find k most similar texts given a query embedding."""
        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return text and similarity score pairs
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.texts):  # Ensure index is valid
                # Convert L2 distance to similarity score (1 / (1 + distance))
                similarity = 1 / (1 + dist)
                results.append((self.texts[idx], similarity))
        
        return results
