"""Semantic embedding module using SentenceTransformers with on-disk caching."""

from typing import List, Dict, Optional
import hashlib
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

class NoteEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: Optional[Path] = None):
        """Initialize the embedder with specified model and optional cache dir."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        self.cache_dir = Path(cache_dir) if cache_dir else Path('.emb_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_text(self, text: str) -> str:
        m = hashlib.sha256()
        m.update(self.model_name.encode('utf-8'))
        m.update(b"\n")
        m.update(text.encode('utf-8'))
        return m.hexdigest()

    def _embedding_path(self, text_hash: str) -> Path:
        return self.cache_dir / f"{text_hash}.npy"

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of text chunks with caching."""
        hashes: List[str] = [self._hash_text(t) for t in texts]
        paths: List[Path] = [self._embedding_path(h) for h in hashes]

        loaded_vectors: Dict[int, np.ndarray] = {}
        to_compute: List[int] = []
        for idx, p in enumerate(paths):
            if p.exists():
                try:
                    loaded_vectors[idx] = np.load(p)
                except Exception:
                    # Corrupt cache entry: recompute
                    to_compute.append(idx)
            else:
                to_compute.append(idx)

        if to_compute:
            texts_to_compute = [texts[i] for i in to_compute]
            computed = self.model.encode(
                texts_to_compute,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            for local_idx, global_idx in enumerate(to_compute):
                vec = np.asarray(computed[local_idx], dtype=np.float32)
                loaded_vectors[global_idx] = vec
                np.save(paths[global_idx], vec)

        # Assemble in original order
        ordered = [loaded_vectors[i] for i in range(len(texts))]
        return np.vstack(ordered).astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text chunk with caching."""
        h = self._hash_text(text)
        p = self._embedding_path(h)
        if p.exists():
            try:
                return np.load(p)
            except Exception:
                pass
        vec = self.model.encode(text, normalize_embeddings=True)
        vec = np.asarray(vec, dtype=np.float32)
        np.save(p, vec)
        return vec
