"""Text summarization module using BART with citation enforcement and optional NLI verification."""

from typing import List, Tuple, Optional, Dict, Union
import re
from transformers import BartForConditionalGeneration, BartTokenizer, AutoTokenizer, AutoModelForSequenceClassification
import torch
from dataclasses import dataclass

@dataclass
class CitationData:
    """Structured citation data for clickable citations."""
    text: str
    full_text: str
    index: Optional[int] = None
    url: Optional[str] = None  # Web URL if available

@dataclass
class SummaryWithCitations:
    """Structured summary with citation data."""
    summary: str
    citation1: CitationData
    citation2: CitationData
    grounded_score: Optional[float] = None
    is_grounded: Optional[bool] = None

class NoteSummarizer:
    def __init__(self, model_name: str = 'facebook/bart-large-cnn', nli_model: Optional[str] = 'facebook/bart-large-mnli'):
        """Initialize the summarizer with specified model and optional NLI verifier."""
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.max_length = 1024
        self.min_length = 40
        self.summary_max_length = 150
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Optional NLI verifier
        self.nli_tokenizer = None
        self.nli_model = None
        if nli_model:
            self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model).to(self.device)

    def summarize_relationship(self, text1: str, text2: str) -> str:
        """Summarize the relationship between two text chunks."""
        # Combine texts with an instruction to produce grounded, concise summary
        prompt = (
            "You are a faithful summarizer. Summarize the relationship between the two passages. "
            "Be concise (2-3 sentences). Avoid speculation."
            "\n\nText 1:\n" + text1 + "\n\nText 2:\n" + text2 + "\n\nRelationship:"
        )
        
        # Tokenize and generate summary
        inputs = self.tokenizer(prompt, max_length=self.max_length, truncation=True, return_tensors="pt").to(self.device)
        
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

    def _extract_citation_spans(self, text: str, max_chars: int = 160) -> str:
        # Heuristic: take the most informative sentence (longest, trimmed)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        if not sentences:
            return text[:max_chars]
        best = max(sentences, key=len)[:max_chars]
        return best
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        return urls

    def _nli_entailment(self, premise: str, hypothesis: str) -> float:
        if not self.nli_model or not self.nli_tokenizer:
            return 1.0
        inputs = self.nli_tokenizer(premise, hypothesis, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        # entailment index = 2 for MNLI family
        return float(probs[0, 2].item())

    def summarize_with_citations(self, text1: str, text2: str, verify: bool = True, 
                                  note1_index: Optional[int] = None, 
                                  note2_index: Optional[int] = None,
                                  return_structured: bool = False) -> Union[str, SummaryWithCitations]:
        """Summarize with citations. Returns structured data if return_structured=True."""
        summary = self.summarize_relationship(text1, text2)
        cite1 = self._extract_citation_spans(text1)
        cite2 = self._extract_citation_spans(text2)
        
        if return_structured:
            # Extract URLs from texts
            urls1 = self._extract_urls(text1)
            urls2 = self._extract_urls(text2)
            url1 = urls1[0] if urls1 else None
            url2 = urls2[0] if urls2 else None
            
            citation1 = CitationData(text=cite1, full_text=text1, index=note1_index, url=url1)
            citation2 = CitationData(text=cite2, full_text=text2, index=note2_index, url=url2)
            
            grounded_score = None
            is_grounded = None
            if verify:
                sentences = re.split(r'(?<=[.!?])\s+', summary)
                sentences = [s.strip() for s in sentences if s.strip()]
                entail_scores = [
                    max(self._nli_entailment(text1, s), self._nli_entailment(text2, s)) for s in sentences
                ]
                grounded = sum(1 for s in entail_scores if s >= 0.5)
                total = max(1, len(entail_scores))
                grounded_score = grounded / total
                is_grounded = grounded_score >= 0.7
            
            return SummaryWithCitations(
                summary=summary,
                citation1=citation1,
                citation2=citation2,
                grounded_score=grounded_score,
                is_grounded=is_grounded
            )
        
        footer = f"\n\nCitations:\n• Text 1: \"{cite1}\"\n• Text 2: \"{cite2}\""
        if verify:
            sentences = re.split(r'(?<=[.!?])\s+', summary)
            sentences = [s.strip() for s in sentences if s.strip()]
            entail_scores = [
                max(self._nli_entailment(text1, s), self._nli_entailment(text2, s)) for s in sentences
            ]
            grounded = sum(1 for s in entail_scores if s >= 0.5)
            total = max(1, len(entail_scores))
            score = grounded / total
            badge = "✅ Grounded" if score >= 0.7 else "⚠ Possibly ungrounded"
            return f"{summary} \n\n[{badge} ({score:.0%} entailed)]{footer}"
        return summary + footer

    def summarize_connections(self, connections: List[Tuple[str, str, float]], verify: bool = True) -> List[str]:
        """Summarize a list of text connections with citations and optional verification."""
        summaries = []
        for text1, text2, _ in connections:
            summary = self.summarize_with_citations(text1, text2, verify=verify)
            summaries.append(summary)
        return summaries
