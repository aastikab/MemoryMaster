"""Core implementation of the Cognitive Twin using local models."""

from pathlib import Path
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import faiss

class CognitiveTwin:
    def __init__(self):
        """Initialize models and vector store."""
        # Initialize SentenceTransformer for embeddings
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize BART-MNLI for text classification/similarity
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        self.classifier = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
        
        # Initialize FAISS index
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.notes = []
        
        # Move models to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(self.device)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts using BART-MNLI."""
        # Prepare the input
        premise = text1
        hypothesis = text2
        
        # Tokenize
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            
        # Return the entailment score as similarity
        return scores[0][2].item()  # Index 2 corresponds to entailment

    def process_notes(self, notes_dir: Path) -> int:
        """Process markdown notes from directory."""
        # Read and clean notes
        for note_path in notes_dir.glob("*.md"):
            with open(note_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    self.notes.append(content)

        if not self.notes:
            return 0

        # Compute embeddings and add to index
        embeddings = self.embedder.encode(
            self.notes,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        self.index.add(np.array(embeddings, dtype=np.float32))
        
        return len(self.notes)

    def find_connections(self, k: int = 5) -> List[str]:
        """Find and summarize connections between notes."""
        summaries = []
        seen_pairs = set()

        # Get embeddings for all notes at once
        embeddings = self.embedder.encode(
            self.notes,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        for i, note in enumerate(self.notes):
            # Find similar notes using FAISS
            D, I = self.index.search(
                np.array([embeddings[i]], dtype=np.float32),
                k=k+1
            )
            
            # Process similar notes
            for j in I[0][1:]:  # Skip the first one (itself)
                if j >= len(self.notes):  # Skip invalid indices
                    continue
                    
                # Create unique pair identifier
                pair = tuple(sorted([i, j]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                
                # Compute semantic similarity
                similarity = self.compute_similarity(self.notes[i], self.notes[j])
                
                # Only include if similarity is high enough
                if similarity > 0.7:
                    # Create a summary highlighting the connection
                    summary = f"Connection found (similarity: {similarity:.2f}):\n"
                    summary += f"Note 1: {self.notes[i][:100]}...\n"
                    summary += f"Note 2: {self.notes[j][:100]}..."
                    summaries.append(summary)
                
                # Limit to top 10 connections
                if len(summaries) >= 10:
                    return summaries
                    
        return summaries