"""Semantic embedding module using SentenceTransformers."""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class NoteEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedder with specified model."""
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of text chunks."""
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text chunk."""
        return self.model.encode(text, normalize_embeddings=True)
