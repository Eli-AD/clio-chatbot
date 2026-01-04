"""Working Memory - Current context window and active conversation state."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import BaseMemory, MemoryEntry, MemoryType, EmotionalValence


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    emotional_tone: Optional[EmotionalValence] = None
    topics: List[str] = field(default_factory=list)


@dataclass
class EmotionalState:
    """Current emotional state tracking."""
    valence: EmotionalValence = EmotionalValence.NEUTRAL
    intensity: float = 0.0  # 0.0 to 1.0
    dominant_emotion: str = "neutral"
    recent_triggers: List[str] = field(default_factory=list)

    def update(self, valence: EmotionalValence, intensity: float, trigger: str = None):
        """Update emotional state with smoothing."""
        # Smooth transition (don't jump instantly)
        self.intensity = (self.intensity * 0.3) + (intensity * 0.7)
        self.valence = valence

        if trigger:
            self.recent_triggers.append(trigger)
            # Keep only last 5 triggers
            self.recent_triggers = self.recent_triggers[-5:]


class WorkingMemory:
    """
    Working Memory - The active context window.

    This is NOT persisted to ChromaDB - it's ephemeral and exists only
    during a session. It holds:
    - Current conversation history
    - Recently retrieved memories from other stores
    - Current emotional state
    - Active context and focus
    """

    def __init__(self, max_turns: int = 20, max_retrieved: int = 10):
        self.max_turns = max_turns
        self.max_retrieved = max_retrieved

        # Conversation state
        self.conversation: List[ConversationTurn] = []
        self.session_start: datetime = datetime.now()

        # Retrieved memories from other stores (kept in working memory for context)
        self.retrieved_memories: List[MemoryEntry] = []

        # Current emotional state
        self.emotional_state = EmotionalState()

        # Active focus/topics
        self.active_topics: List[str] = []
        self.current_focus: Optional[str] = None

        # Context flags
        self.context: Dict[str, Any] = {
            "is_first_session": False,
            "time_since_last": None,
            "user_mood_hint": None,
            "conversation_depth": 0,  # How deep into a topic we are
        }

    def add_turn(
        self,
        role: str,
        content: str,
        emotional_tone: EmotionalValence = None,
        topics: List[str] = None
    ):
        """Add a conversation turn."""
        turn = ConversationTurn(
            role=role,
            content=content,
            emotional_tone=emotional_tone,
            topics=topics or []
        )
        self.conversation.append(turn)

        # Update active topics
        if topics:
            for topic in topics:
                if topic not in self.active_topics:
                    self.active_topics.append(topic)
            # Keep only recent topics
            self.active_topics = self.active_topics[-10:]

        # Trim conversation if too long
        if len(self.conversation) > self.max_turns:
            self.conversation = self.conversation[-self.max_turns:]

        # Update conversation depth
        self.context["conversation_depth"] = len(self.conversation)

    def add_retrieved_memory(self, memory: MemoryEntry):
        """Add a memory retrieved from another store to working context."""
        # Avoid duplicates
        if any(m.id == memory.id for m in self.retrieved_memories):
            return

        self.retrieved_memories.append(memory)

        # Trim if too many
        if len(self.retrieved_memories) > self.max_retrieved:
            # Remove least important
            self.retrieved_memories.sort(key=lambda m: m.get_effective_importance(), reverse=True)
            self.retrieved_memories = self.retrieved_memories[:self.max_retrieved]

    def get_conversation_history(self, last_n: int = None) -> List[Dict[str, str]]:
        """Get conversation in chat format for LLM."""
        turns = self.conversation[-last_n:] if last_n else self.conversation
        return [{"role": t.role, "content": t.content} for t in turns]

    def get_context_summary(self) -> str:
        """Get a summary of current working memory context."""
        parts = []

        # Emotional state
        if self.emotional_state.intensity > 0.3:
            parts.append(f"Current mood: {self.emotional_state.valence.value} "
                        f"(intensity: {self.emotional_state.intensity:.1f})")

        # Active topics
        if self.active_topics:
            parts.append(f"Active topics: {', '.join(self.active_topics[-5:])}")

        # Current focus
        if self.current_focus:
            parts.append(f"Current focus: {self.current_focus}")

        # Retrieved memories summary
        if self.retrieved_memories:
            parts.append(f"Relevant memories loaded: {len(self.retrieved_memories)}")

        return " | ".join(parts) if parts else "Fresh conversation"

    def get_relevant_retrieved(self, query: str, n: int = 3) -> List[MemoryEntry]:
        """Get retrieved memories most relevant to current query."""
        if not self.retrieved_memories:
            return []

        # Simple relevance: check for word overlap
        query_words = set(query.lower().split())
        scored = []

        for mem in self.retrieved_memories:
            mem_words = set(mem.content.lower().split())
            overlap = len(query_words & mem_words)
            importance = mem.get_effective_importance()
            score = (overlap * 0.5) + (importance * 0.5)
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:n]]

    def update_emotional_state(
        self,
        valence: EmotionalValence,
        intensity: float,
        trigger: str = None
    ):
        """Update the current emotional state."""
        self.emotional_state.update(valence, intensity, trigger)

    def set_context(self, key: str, value: Any):
        """Set a context value."""
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self.context.get(key, default)

    def get_session_duration(self) -> float:
        """Get session duration in minutes."""
        delta = datetime.now() - self.session_start
        return delta.total_seconds() / 60

    def get_word_count(self) -> int:
        """Get total words in conversation."""
        return sum(len(t.content.split()) for t in self.conversation)

    def clear(self):
        """Clear working memory (end of session)."""
        self.conversation = []
        self.retrieved_memories = []
        self.active_topics = []
        self.current_focus = None
        self.emotional_state = EmotionalState()
        self.session_start = datetime.now()
        self.context = {
            "is_first_session": False,
            "time_since_last": None,
            "user_mood_hint": None,
            "conversation_depth": 0,
        }

    def to_dict(self) -> dict:
        """Serialize working memory state (for debugging/logging)."""
        return {
            "session_start": self.session_start.isoformat(),
            "conversation_turns": len(self.conversation),
            "retrieved_memories": len(self.retrieved_memories),
            "emotional_state": {
                "valence": self.emotional_state.valence.value,
                "intensity": self.emotional_state.intensity,
            },
            "active_topics": self.active_topics,
            "current_focus": self.current_focus,
            "context": self.context,
        }
