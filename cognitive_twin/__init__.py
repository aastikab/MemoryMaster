"""
Cognitive Twin - A system for semantic note processing and relationship discovery.
"""

from .core import CognitiveTwin
from .embedder import NoteEmbedder
from .summarizer import NoteSummarizer, SummaryWithCitations, CitationData
from .vector_store import VectorStore
from .exporter import SummaryExporter
from .visualizer import NoteVisualizer
from .chatbot import ChatbotManager, OpenAIChatbot, ClaudeChatbot

__version__ = "0.1.0"

__all__ = [
    'CognitiveTwin',
    'NoteEmbedder',
    'NoteSummarizer',
    'SummaryWithCitations',
    'CitationData',
    'VectorStore',
    'SummaryExporter',
    'NoteVisualizer',
    'ChatbotManager',
    'OpenAIChatbot',
    'ClaudeChatbot',
]
