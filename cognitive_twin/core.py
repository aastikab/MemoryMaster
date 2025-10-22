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
        print(f"🔍 Computing similarity between texts...")
        print(f"📝 Text 1: {text1[:100]}...")
        print(f"📝 Text 2: {text2[:100]}...")
        
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
        print(f"🔤 Tokenized input shape: {inputs['input_ids'].shape}")
        
        # Get prediction
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            print(f"📊 Raw scores: {outputs.logits}")
            print(f"📊 Softmax scores: {scores}")
            
        # Return the entailment score as similarity
        entailment_score = scores[0][2].item()  # Index 2 corresponds to entailment
        print(f"🎯 Entailment score: {entailment_score:.4f}")
        return entailment_score

    def process_notes(self, notes_dir: Path) -> int:
        """Process markdown notes from directory."""
        print(f"🔍 Processing notes from: {notes_dir}")
        
        # Read and clean notes
        for note_path in notes_dir.glob("*.md"):
            print(f"📄 Reading file: {note_path}")
            with open(note_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    self.notes.append(content)
                    print(f"✅ Added note ({len(content)} chars): {content[:100]}...")

        if not self.notes:
            print("❌ No notes found!")
            return 0

        print(f"📊 Total notes loaded: {len(self.notes)}")
        print("🧠 Computing embeddings...")

        # Compute embeddings and add to index
        embeddings = self.embedder.encode(
            self.notes,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        print(f"📐 Embeddings shape: {embeddings.shape}")
        self.index.add(np.array(embeddings, dtype=np.float32))
        print(f"🎯 Index size: {self.index.ntotal}")
        
        return len(self.notes)

    def find_connections(self, k: int = 5, threshold: float = 0.6) -> List[str]:
        """Find and summarize connections between notes.
        Args:
            k: number of neighbors to retrieve per note
            threshold: entailment similarity threshold to include a pair
        """
        print(f"🔍 Finding connections between {len(self.notes)} notes...")
        summaries = []
        seen_pairs = set()

        # Get embeddings for all notes at once
        print("🧠 Computing embeddings for similarity search...")
        embeddings = self.embedder.encode(
            self.notes,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        print(f"📐 Embeddings shape: {embeddings.shape}")

        for i, note in enumerate(self.notes):
            print(f"🔍 Processing note {i+1}/{len(self.notes)}")
            
            # Find similar notes using FAISS
            D, I = self.index.search(
                np.array([embeddings[i]], dtype=np.float32),
                k=k+1
            )
            print(f"📊 Found {len(I[0])} similar notes: {I[0]}")
            print(f"📏 Distances: {D[0]}")
            
            # Process similar notes
            for j in I[0][1:]:  # Skip the first one (itself)
                if j >= len(self.notes):  # Skip invalid indices
                    print(f"⚠️ Skipping invalid index: {j}")
                    continue
                    
                # Create unique pair identifier
                pair = tuple(sorted([i, j]))
                if pair in seen_pairs:
                    print(f"🔄 Skipping already processed pair: {pair}")
                    continue
                seen_pairs.add(pair)
                
                print(f"🔗 Comparing notes {i} and {j}")
                print(f"📝 Note {i}: {self.notes[i][:50]}...")
                print(f"📝 Note {j}: {self.notes[j][:50]}...")
                
                # Compute semantic similarity
                similarity = self.compute_similarity(self.notes[i], self.notes[j])
                print(f"🎯 Similarity score: {similarity:.4f}")
                
                # Only include if similarity is high enough
                if similarity >= threshold:
                    print(f"✅ High similarity found! ({similarity:.4f})")
                    # Create a summary highlighting the connection
                    summary = f"Connection found (similarity: {similarity:.2f}):\n"
                    summary += f"Note 1: {self.notes[i][:100]}...\n"
                    summary += f"Note 2: {self.notes[j][:100]}..."
                    summaries.append(summary)
                else:
                    print(f"❌ Low similarity: {similarity:.4f} (threshold: {threshold})")
                
                # Limit to top 10 connections
                if len(summaries) >= 10:
                    print(f"🛑 Reached limit of 10 connections")
                    return summaries
                    
        print(f"📊 Total connections found: {len(summaries)}")
        return summaries