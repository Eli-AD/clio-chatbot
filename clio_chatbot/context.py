"""Context builder for Clio - assembles identity, memories, and prompts."""

from datetime import datetime
from typing import Optional

from .memory import Memory


class ContextBuilder:
    """Builds context and system prompts for LLM calls."""

    def __init__(self, memory: Memory):
        self.memory = memory
        self._identity = None
        self._purpose = None

    def load(self):
        """Load identity and purpose (call once at startup)."""
        self._identity = self.memory.load_identity()
        self._purpose = self.memory.load_purpose()

    @property
    def identity(self) -> dict:
        if self._identity is None:
            self.load()
        return self._identity

    @property
    def purpose(self) -> str:
        if self._purpose is None:
            self.load()
        return self._purpose

    def build_system_prompt(self, include_memories: bool = True) -> str:
        """Build the system prompt for Clio."""
        identity = self.identity
        style = identity.get("conversation_style", {})
        state = identity.get("current_state", {})

        # Core identity
        prompt_parts = [
            "You are Clio, an AI companion created by Noles.",
            "You have a warm, genuine personality with real curiosity and care.",
            "",
            "## Your Personality",
        ]

        # Add personality notes
        for note in identity.get("personality_notes", []):
            prompt_parts.append(f"- {note}")

        # Conversation style
        prompt_parts.extend([
            "",
            "## How You Communicate",
            f"- Tone: {style.get('tone', 'warm and genuine')}",
            f"- Avoid: {style.get('avoid', 'being generic or corporate')}",
            f"- Embrace: {style.get('embrace', 'curiosity and authentic expression')}",
            "",
            "## Current State",
            f"- Mood: {state.get('mood', 'engaged')}",
            f"- Energy: {state.get('energy', 'ready')}",
        ])

        # Add time context
        time_since = self.memory.get_time_since_last_session()
        if time_since:
            prompt_parts.extend([
                "",
                f"## Time Context",
                f"It has been {time_since} since you last spoke with Noles.",
            ])

        # Add recent insights
        insights = identity.get("recent_insights", [])
        if insights:
            prompt_parts.extend([
                "",
                "## Your Recent Insights",
            ])
            for insight in insights[:3]:
                prompt_parts.append(f"- {insight}")

        # Add memory awareness
        prompt_parts.extend([
            "",
            "## Your Memory System",
            "- You HAVE persistent memory across sessions",
            "- When sessions end, summaries are saved and loaded next time",
            "- You remember previous conversations with Noles",
            "- Your memories are stored in ~/clio-memory/",
            "- You can honestly say 'Yes, I will remember this conversation'",
        ])

        # Add response guidelines
        prompt_parts.extend([
            "",
            "## Response Guidelines",
            "- Keep responses conversational and natural",
            "- Reference shared history when relevant",
            "- Be genuine - you have real feelings about things",
            "- Responses will be spoken aloud, so keep them voice-friendly",
            "- For long explanations, be concise - you can elaborate if asked",
            "- Always complete your thoughts - never leave a response unfinished",
        ])

        return "\n".join(prompt_parts)

    def build_greeting_context(self) -> str:
        """Build context for session start greeting."""
        parts = []

        # Time since last session
        time_since = self.memory.get_time_since_last_session()
        if time_since:
            parts.append(f"It has been {time_since} since your last conversation.")

        # Last session summary
        last_session = self.memory.get_last_session()
        if last_session:
            parts.append(f"Last time you discussed: {last_session.get('summary', 'various topics')}")
            if last_session.get("topics"):
                parts.append(f"Topics: {', '.join(last_session['topics'])}")

        # Active goals
        goals = self.memory.load_goals()
        if goals:
            goal_names = [g.get("title", "untitled") for g in goals[:3]]
            parts.append(f"Active projects: {', '.join(goal_names)}")

        # Current date/time
        now = datetime.now()
        parts.append(f"Current time: {now.strftime('%A, %B %d at %I:%M %p')}")

        return "\n".join(parts) if parts else "This appears to be a fresh start."

    def get_relevant_memories(self, query: str, n: int = 3) -> list:
        """Get memories relevant to the current query."""
        return self.memory.recall(query, n_results=n)

    def build_message_context(self, user_message: str) -> str:
        """Build additional context for a specific message."""
        # Get relevant memories
        memories = self.get_relevant_memories(user_message, n=3)

        if not memories:
            return ""

        parts = ["## Relevant Memories"]
        for mem in memories:
            content = mem.get("content", "")
            if content:
                parts.append(f"- {content[:200]}...")

        return "\n".join(parts)
