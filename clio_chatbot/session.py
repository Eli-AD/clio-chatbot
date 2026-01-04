"""Session lifecycle management for Clio."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .memory import Memory


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    backend: str = "unknown"  # Which LLM generated this


@dataclass
class Session:
    """Represents a conversation session."""
    id: str
    started_at: datetime
    messages: List[Message] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    mood: str = "neutral"
    is_active: bool = True

    def add_message(self, role: str, content: str, backend: str = "unknown"):
        self.messages.append(Message(
            role=role,
            content=content,
            backend=backend
        ))

    def get_conversation_history(self, last_n: int = 10) -> List[dict]:
        """Get recent messages in chat format."""
        recent = self.messages[-last_n:] if len(self.messages) > last_n else self.messages
        return [{"role": m.role, "content": m.content} for m in recent]

    def get_word_count(self) -> int:
        """Total words in conversation."""
        return sum(len(m.content.split()) for m in self.messages)


class SessionManager:
    """Manages session lifecycle."""

    def __init__(self, memory: Memory):
        self.memory = memory
        self.current_session: Optional[Session] = None
        self.idle_timeout_seconds = 300  # 5 minutes

    def start_session(self) -> Session:
        """Start a new session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session = Session(
            id=session_id,
            started_at=datetime.now()
        )
        return self.current_session

    def end_session(self, summary: str = None):
        """End the current session and save it."""
        if not self.current_session:
            return

        session = self.current_session

        # Generate summary if not provided
        if not summary:
            summary = self._generate_summary(session)

        # Extract topics from conversation
        topics = self._extract_topics(session)

        # Save to memory
        self.memory.save_session(
            summary=summary,
            topics=topics,
            mood=session.mood
        )

        # Update shared state for daemon
        self.memory.update_shared_state("last_conversation", {
            "ended_at": datetime.now().isoformat(),
            "summary": summary,
            "topics": topics,
            "message_count": len(session.messages)
        })

        session.is_active = False
        self.current_session = None

    def _generate_summary(self, session: Session) -> str:
        """Generate a simple summary of the session."""
        if not session.messages:
            return "No messages exchanged."

        # Simple approach: summarize based on message count and content
        msg_count = len(session.messages)
        user_msgs = [m for m in session.messages if m.role == "user"]

        if msg_count <= 2:
            return "Brief exchange."

        # Get first and last user messages as summary anchors
        first_topic = user_msgs[0].content[:100] if user_msgs else "general chat"
        last_topic = user_msgs[-1].content[:100] if len(user_msgs) > 1 else first_topic

        return f"Conversation with {msg_count} messages. Started discussing: {first_topic}"

    def _extract_topics(self, session: Session) -> List[str]:
        """Extract main topics from conversation."""
        # Simple keyword extraction
        all_text = " ".join(m.content.lower() for m in session.messages)
        words = all_text.split()

        # Common topic indicators
        topic_words = set()
        for word in words:
            # Skip short words and common words
            if len(word) > 4 and word.isalpha():
                topic_words.add(word)

        # Return top topics by frequency (simplified)
        from collections import Counter
        word_counts = Counter(word for word in words if len(word) > 4 and word.isalpha())
        return [word for word, _ in word_counts.most_common(5)]

    def add_message(self, role: str, content: str, backend: str = "unknown"):
        """Add a message to the current session."""
        if self.current_session:
            self.current_session.add_message(role, content, backend)

    def get_history(self, last_n: int = 10) -> List[dict]:
        """Get recent conversation history."""
        if self.current_session:
            return self.current_session.get_conversation_history(last_n)
        return []

    def get_greeting_info(self) -> dict:
        """Get information for session greeting."""
        last_session = self.memory.get_last_session()
        time_since = self.memory.get_time_since_last_session()

        return {
            "time_since": time_since,
            "last_summary": last_session.get("summary") if last_session else None,
            "last_topics": last_session.get("topics", []) if last_session else [],
            "is_first_session": last_session is None
        }
