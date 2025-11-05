"""
Cognitive Twin - A system for semantic note processing and relationship discovery.
"""

from .core import CognitiveTwin
from .embedder import NoteEmbedder
from .summarizer import NoteSummarizer
from .vector_store import VectorStore

__version__ = "0.1.0"

__all__ = [
    'CognitiveTwin',
    'NoteEmbedder',
    'NoteSummarizer',
    'VectorStore',
]
