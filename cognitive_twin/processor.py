"""Note processor module for cleaning and preprocessing notes."""

import re
from pathlib import Path
from typing import List, Union

class NoteProcessor:
    def __init__(self):
        self.min_chunk_size = 100  # minimum characters per chunk
        self.max_chunk_size = 512  # maximum characters per chunk

    def load_note(self, note_path: Union[str, Path]) -> str:
        """Load note content from file."""
        with open(note_path, 'r', encoding='utf-8') as f:
            return f.read()

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into semantic chunks for processing."""
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # Filter out chunks that are too small
        return [chunk for chunk in chunks if len(chunk) >= self.min_chunk_size]
