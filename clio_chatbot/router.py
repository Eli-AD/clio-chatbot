"""Hybrid LLM router - routes queries to appropriate backend."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LLMBackend(Enum):
    OLLAMA_SMALL = "qwen2.5:1.5b"
    OLLAMA_LARGE = "qwen2.5:7b"
    CLAUDE = "claude"


@dataclass
class RoutingDecision:
    backend: LLMBackend
    reason: str
    confidence: float


# Words that suggest emotional/relationship content
EMOTIONAL_WORDS = {
    "feel", "feeling", "feelings", "happy", "sad", "angry", "frustrated",
    "love", "hate", "miss", "worried", "anxious", "excited", "scared",
    "hurt", "lonely", "grateful", "proud", "ashamed", "confused",
    "relationship", "friendship", "trust", "care", "appreciate"
}

# Words that suggest complex reasoning
REASONING_WORDS = {
    "why", "how", "explain", "analyze", "compare", "contrast",
    "think", "reason", "logic", "because", "therefore", "however",
    "implications", "consequences", "tradeoffs", "strategy",
    "design", "architect", "plan", "approach"
}

# Code indicators
CODE_PATTERNS = [
    r'```',
    r'def\s+\w+',
    r'class\s+\w+',
    r'import\s+\w+',
    r'function\s+\w+',
    r'\bcode\b',
    r'\bbug\b',
    r'\berror\b',
    r'\bfix\b.*\bcode\b',
    r'\bwrite\b.*\b(script|program|function)\b',
]

# Greeting patterns
GREETING_PATTERNS = [
    r'^(hi|hello|hey|yo|sup|greetings|good\s+(morning|afternoon|evening))[\s!.,?]*$',
    r"^what'?s\s+up[\s!?,]*$",
    r'^how\s+are\s+you[\s!?,]*$',
]

# Escalation phrases
ESCALATION_PHRASES = [
    "think carefully",
    "think deeply",
    "step by step",
    "be thorough",
    "important",
    "critical",
    "complex",
    "difficult",
]


class Router:
    """Routes queries to the appropriate LLM backend."""

    def __init__(self):
        self.code_patterns = [re.compile(p, re.IGNORECASE) for p in CODE_PATTERNS]
        self.greeting_patterns = [re.compile(p, re.IGNORECASE) for p in GREETING_PATTERNS]

    def route(self, query: str, conversation_depth: int = 0) -> RoutingDecision:
        """Determine which LLM to use for this query."""
        # Always use Claude for best quality and memory tool support
        return RoutingDecision(
            backend=LLMBackend.CLAUDE,
            reason="Claude-only mode",
            confidence=1.0
        )

        # --- Original routing logic (disabled) ---
        query_lower = query.lower()
        words = set(query_lower.split())
        word_count = len(query.split())

        # Check for greetings (fast path)
        for pattern in self.greeting_patterns:
            if pattern.match(query.strip()):
                return RoutingDecision(
                    backend=LLMBackend.OLLAMA_SMALL,
                    reason="Simple greeting",
                    confidence=0.95
                )

        # Check for code-related content -> Claude
        for pattern in self.code_patterns:
            if pattern.search(query):
                return RoutingDecision(
                    backend=LLMBackend.CLAUDE,
                    reason="Code-related query",
                    confidence=0.9
                )

        # Check for explicit escalation phrases -> Claude
        for phrase in ESCALATION_PHRASES:
            if phrase in query_lower:
                return RoutingDecision(
                    backend=LLMBackend.CLAUDE,
                    reason=f"Escalation phrase: '{phrase}'",
                    confidence=0.85
                )

        # Check for emotional content -> Claude (relationships matter)
        emotional_overlap = words & EMOTIONAL_WORDS
        if len(emotional_overlap) >= 2:
            return RoutingDecision(
                backend=LLMBackend.CLAUDE,
                reason="Emotional/relationship content",
                confidence=0.8
            )

        # Check for complex reasoning -> Claude
        reasoning_overlap = words & REASONING_WORDS
        if len(reasoning_overlap) >= 2:
            return RoutingDecision(
                backend=LLMBackend.CLAUDE,
                reason="Complex reasoning required",
                confidence=0.75
            )

        # Length-based routing
        if word_count < 10:
            return RoutingDecision(
                backend=LLMBackend.OLLAMA_SMALL,
                reason="Short query",
                confidence=0.7
            )
        elif word_count < 40:
            return RoutingDecision(
                backend=LLMBackend.OLLAMA_LARGE,
                reason="Medium-length query",
                confidence=0.7
            )
        else:
            return RoutingDecision(
                backend=LLMBackend.CLAUDE,
                reason="Long/complex query",
                confidence=0.65
            )

    def should_use_claude(self, query: str) -> bool:
        """Simple check if Claude should be used."""
        decision = self.route(query)
        return decision.backend == LLMBackend.CLAUDE

    def get_model_name(self, query: str) -> str:
        """Get the model name string for the query."""
        decision = self.route(query)
        return decision.backend.value
