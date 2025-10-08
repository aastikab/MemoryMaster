"""Text summarization module using BART."""

from typing import List, Tuple
from transformers import BartForConditionalGeneration, BartTokenizer

class NoteSummarizer:
    def __init__(self, model_name: str = 'facebook/bart-large-cnn'):
        """Initialize the summarizer with specified model."""
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.max_length = 1024
        self.min_length = 40
        self.summary_max_length = 150

    def summarize_relationship(self, text1: str, text2: str) -> str:
        """Summarize the relationship between two text chunks."""
        # Combine texts with a separator
        combined = f"Text 1: {text1}\nText 2: {text2}\nRelationship:"
        
        # Tokenize and generate summary
        inputs = self.tokenizer(combined, max_length=self.max_length, truncation=True, return_tensors="pt")
        
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=self.summary_max_length,
            min_length=self.min_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def summarize_connections(self, connections: List[Tuple[str, str, float]]) -> List[str]:
        """Summarize a list of text connections."""
        summaries = []
        for text1, text2, _ in connections:
            summary = self.summarize_relationship(text1, text2)
            summaries.append(summary)
        return summaries
