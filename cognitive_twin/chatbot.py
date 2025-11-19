"""Chatbot integration module with OpenAI and Claude support."""

from typing import List, Dict, Optional, Union
import os
from abc import ABC, abstractmethod

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class ChatbotProvider(ABC):
    """Abstract base class for chatbot providers."""
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], context: Optional[str] = None) -> str:
        """Send chat messages and get response."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass

class OpenAIChatbot(ChatbotProvider):
    """OpenAI ChatGPT integration."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """Initialize OpenAI chatbot.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Model to use (gpt-3.5-turbo, gpt-4, etc.)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")
        
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def is_available(self) -> bool:
        return OPENAI_AVAILABLE and self.api_key is not None
    
    def chat(self, messages: List[Dict[str, str]], context: Optional[str] = None) -> str:
        """Send chat messages with optional context injection.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            context: Optional context string to inject into system message
        
        Returns:
            Assistant's response text
        """
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant for analyzing and connecting notes."
        }
        
        if context:
            system_message["content"] += f"\n\nContext from current session:\n{context}"
        
        full_messages = [system_message] + messages
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages
        )
        
        return response.choices[0].message.content

class ClaudeChatbot(ChatbotProvider):
    """Anthropic Claude integration."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        """Initialize Claude chatbot.
        
        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
            model: Model to use (claude-3-opus, claude-3-sonnet, claude-3-haiku)
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key.")
        
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def is_available(self) -> bool:
        return ANTHROPIC_AVAILABLE and self.api_key is not None
    
    def chat(self, messages: List[Dict[str, str]], context: Optional[str] = None) -> str:
        """Send chat messages with optional context injection.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            context: Optional context string to inject
        
        Returns:
            Assistant's response text
        """
        system_prompt = "You are a helpful assistant for analyzing and connecting notes."
        
        if context:
            system_prompt += f"\n\nContext from current session:\n{context}"
        
        # Convert messages to Claude format
        claude_messages = []
        for msg in messages:
            if msg["role"] == "user":
                claude_messages.append({
                    "role": "user",
                    "content": msg["content"]
                })
            elif msg["role"] == "assistant":
                claude_messages.append({
                    "role": "assistant",
                    "content": msg["content"]
                })
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=claude_messages
        )
        
        return response.content[0].text

class ChatbotManager:
    """Manager for chatbot providers with privacy guardrails."""
    
    def __init__(self, provider: str = "openai", use_local_summaries: bool = True):
        """Initialize chatbot manager.
        
        Args:
            provider: 'openai' or 'claude'
            use_local_summaries: If True, only use local summaries (privacy guardrail)
        """
        self.provider_name = provider.lower()
        self.use_local_summaries = use_local_summaries
        self.provider: Optional[ChatbotProvider] = None
        self.chat_history: List[Dict[str, str]] = []
        
        if self.provider_name == "openai":
            try:
                self.provider = OpenAIChatbot()
            except (ImportError, ValueError) as e:
                print(f"Warning: OpenAI not available: {e}")
        elif self.provider_name == "claude":
            try:
                self.provider = ClaudeChatbot()
            except (ImportError, ValueError) as e:
                print(f"Warning: Claude not available: {e}")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'claude'")
    
    def is_available(self) -> bool:
        """Check if chatbot is available."""
        return self.provider is not None and self.provider.is_available()
    
    def inject_context(self, notes: List[str], connections: List[Dict], 
                      summaries: List[str]) -> str:
        """Create context string from current session.
        
        Args:
            notes: List of note texts
            connections: List of connection dictionaries
            summaries: List of summary strings
        
        Returns:
            Context string for injection
        """
        context_parts = []
        
        if notes:
            context_parts.append(f"Notes in session ({len(notes)}):")
            for i, note in enumerate(notes[:5]):  # Limit to first 5
                preview = note[:200] + "..." if len(note) > 200 else note
                context_parts.append(f"  Note {i}: {preview}")
        
        if connections:
            context_parts.append(f"\nConnections found ({len(connections)}):")
            for i, conn in enumerate(connections[:5]):  # Limit to first 5
                if 'similarity' in conn:
                    context_parts.append(f"  Connection {i}: Similarity {conn['similarity']:.2%}")
        
        if summaries:
            context_parts.append(f"\nSummaries ({len(summaries)}):")
            for i, summary in enumerate(summaries[:3]):  # Limit to first 3
                preview = summary[:200] + "..." if len(summary) > 200 else summary
                context_parts.append(f"  Summary {i}: {preview}")
        
        return "\n".join(context_parts)
    
    def chat(self, user_message: str, context: Optional[str] = None) -> str:
        """Send a chat message.
        
        Args:
            user_message: User's message
            context: Optional context to inject
        
        Returns:
            Assistant's response
        """
        if not self.is_available():
            return "Chatbot not available. Please configure API keys."
        
        # Add user message to history
        self.chat_history.append({"role": "user", "content": user_message})
        
        # Get response
        response = self.provider.chat(self.chat_history, context=context)
        
        # Add assistant response to history
        self.chat_history.append({"role": "assistant", "content": response})
        
        return response
    
    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
    
    def get_summary_local_only(self, text: str, local_summarizer) -> str:
        """Get summary using local model only (privacy guardrail).
        
        Args:
            text: Text to summarize
            local_summarizer: Local summarizer instance
        
        Returns:
            Summary text
        """
        if self.use_local_summaries:
            # Use local summarizer instead of API
            # This is a placeholder - you'd integrate with your local summarizer
            return "Local summary (privacy-preserved)"
        else:
            # Fall back to API if allowed
            return self.chat(f"Summarize this text: {text}")

