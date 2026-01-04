"""Memory Manager - Orchestrates all memory types and handles consolidation."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import MemoryEntry, MemoryType, EmotionalValence, MEMORY_DIR
from .working import WorkingMemory, EmotionalState
from .episodic import EpisodicMemory
from .semantic import SemanticMemory, KnowledgeCategory
from .longterm import LongTermMemory, ConsolidationType


class MemoryManager:
    """
    Central orchestrator for Clio's memory system.

    Handles:
    - Memory storage across all types
    - Intelligent recall with cross-memory search
    - Session lifecycle (start/end)
    - Memory consolidation
    - Emotional state tracking
    - Context building for LLM prompts
    """

    def __init__(self):
        # Initialize all memory stores
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.longterm = LongTermMemory()

        # Paths for file-based data
        self.memory_dir = MEMORY_DIR
        self.identity_file = self.memory_dir / "identity.json"
        self.shared_state_file = self.memory_dir / "shared_state.json"

        # Session tracking
        self.session_id: Optional[str] = None
        self.session_start: Optional[datetime] = None

    # =========================================================================
    # SESSION LIFECYCLE
    # =========================================================================

    def start_session(self) -> dict:
        """
        Start a new session and load relevant context.

        Returns session context info for greeting generation.
        """
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()

        # Clear working memory from any previous session
        self.working.clear()

        # Load foundational memories
        foundation = self.longterm.get_session_foundation()

        # Get time since last session
        time_since = self._get_time_since_last()

        # Load recent episodic memories
        recent_episodes = self.episodic.get_recent(n=3)
        for episode in recent_episodes:
            self.working.add_retrieved_memory(episode)

        # Get last session summary
        last_session = self._get_last_session_summary()

        # Set working memory context
        self.working.set_context("is_first_session", time_since is None)
        self.working.set_context("time_since_last", time_since)
        self.working.set_context("last_session", last_session)

        return {
            "session_id": self.session_id,
            "is_first_session": time_since is None,
            "time_since": time_since,
            "last_session": last_session,
            "foundation": foundation,
            "recent_episodes": [e.content for e in recent_episodes],
        }

    def end_session(
        self,
        summary: str = None,
        topics: List[str] = None,
        emotional_summary: Tuple[EmotionalValence, float] = None,
    ):
        """
        End the current session with proper memory consolidation.

        Args:
            summary: Optional summary of the session
            topics: Topics discussed
            emotional_summary: (valence, intensity) of overall session
        """
        if not self.session_id:
            return

        # Generate summary if not provided
        if not summary:
            summary = self._generate_session_summary()

        topics = topics or self.working.active_topics

        # Determine emotional context
        if emotional_summary:
            valence, intensity = emotional_summary
        else:
            valence = self.working.emotional_state.valence
            intensity = self.working.emotional_state.intensity

        # Calculate session duration
        duration = self.working.get_session_duration()

        # Store as episodic memory
        episode = self.episodic.store_conversation_episode(
            summary=summary,
            topics=topics,
            emotional_valence=valence,
            emotional_intensity=intensity,
            key_moments=self._extract_key_moments(),
            duration_minutes=duration,
        )

        # Update shared state for daemon/other systems
        self._update_shared_state({
            "last_conversation": {
                "ended_at": datetime.now().isoformat(),
                "session_id": self.session_id,
                "summary": summary,
                "topics": topics,
                "duration_minutes": duration,
                "emotional_valence": valence.value,
                "message_count": len(self.working.conversation),
            }
        })

        # Run consolidation check
        self._maybe_consolidate()

        # Clear session state
        self.working.clear()
        self.session_id = None
        self.session_start = None

    # =========================================================================
    # MEMORY OPERATIONS
    # =========================================================================

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        importance: float = 0.5,
        emotional_valence: EmotionalValence = EmotionalValence.NEUTRAL,
        emotional_intensity: float = 0.0,
        tags: List[str] = None,
        **kwargs
    ) -> MemoryEntry:
        """
        Store a memory in the appropriate store.

        This is the main entry point for storing memories.
        """
        if memory_type == MemoryType.EPISODIC:
            return self.episodic.store(
                content=content,
                importance=importance,
                emotional_valence=emotional_valence,
                emotional_intensity=emotional_intensity,
                tags=tags,
                **kwargs
            )
        elif memory_type == MemoryType.SEMANTIC:
            category = kwargs.pop("category", KnowledgeCategory.WORLD_KNOWLEDGE)
            return self.semantic.store(
                content=content,
                category=category,
                importance=importance,
                tags=tags,
                **kwargs
            )
        elif memory_type == MemoryType.LONGTERM:
            ctype = kwargs.pop("consolidation_type", ConsolidationType.LESSON_LEARNED)
            return self.longterm.store(
                content=content,
                consolidation_type=ctype,
                importance=importance,
                emotional_valence=emotional_valence,
                emotional_intensity=emotional_intensity,
                tags=tags,
                **kwargs
            )
        else:
            # Default to semantic
            return self.semantic.store(
                content=content,
                importance=importance,
                tags=tags,
            )

    def recall(
        self,
        query: str,
        n_results: int = 5,
        memory_types: List[MemoryType] = None,
        include_working: bool = True,
    ) -> List[MemoryEntry]:
        """
        Recall memories relevant to query across all stores.

        Args:
            query: What to search for
            n_results: Max results total
            memory_types: Which stores to search (None = all)
            include_working: Include working memory's retrieved memories
        """
        results = []

        # Default to all types
        if memory_types is None:
            memory_types = [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.LONGTERM]

        # Search each store
        per_store = max(2, n_results // len(memory_types))

        if MemoryType.LONGTERM in memory_types:
            results.extend(self.longterm.recall(query, n_results=per_store))

        if MemoryType.SEMANTIC in memory_types:
            results.extend(self.semantic.recall(query, n_results=per_store))

        if MemoryType.EPISODIC in memory_types:
            results.extend(self.episodic.recall(query, n_results=per_store))

        # Include relevant items from working memory
        if include_working:
            working_relevant = self.working.get_relevant_retrieved(query, n=2)
            results.extend(working_relevant)

        # Sort by effective importance and deduplicate
        seen_ids = set()
        unique_results = []
        for entry in results:
            if entry.id not in seen_ids:
                seen_ids.add(entry.id)
                unique_results.append(entry)

        unique_results.sort(key=lambda e: e.get_effective_importance(), reverse=True)

        # Add to working memory for context
        for entry in unique_results[:n_results]:
            self.working.add_retrieved_memory(entry)

        return unique_results[:n_results]

    def add_conversation_turn(
        self,
        role: str,
        content: str,
        topics: List[str] = None,
    ):
        """Add a conversation turn to working memory."""
        self.working.add_turn(
            role=role,
            content=content,
            topics=topics,
        )

    # =========================================================================
    # CONTEXT BUILDING
    # =========================================================================

    def build_context_for_message(self, user_message: str) -> str:
        """
        Build relevant context to inject into LLM prompt.

        Searches memories and builds a context section.
        """
        parts = []

        # Get relevant memories
        memories = self.recall(user_message, n_results=5)

        if memories:
            parts.append("## Relevant Memories")
            for mem in memories:
                type_label = mem.memory_type.value.capitalize()
                parts.append(f"[{type_label}] {mem.content[:200]}...")

        # Add emotional context if significant
        if self.working.emotional_state.intensity > 0.3:
            parts.append(f"\n## Current Emotional Context")
            parts.append(f"Mood: {self.working.emotional_state.valence.value}")

        # Add active topics
        if self.working.active_topics:
            parts.append(f"\n## Active Topics: {', '.join(self.working.active_topics[-5:])}")

        return "\n".join(parts) if parts else ""

    def build_system_prompt_additions(self) -> str:
        """
        Build additions to system prompt from long-term memory.

        This gives Clio her sense of identity and continuity.
        """
        return self.longterm.build_identity_prompt()

    def get_conversation_history(self, last_n: int = 10) -> List[Dict[str, str]]:
        """Get conversation history for LLM."""
        return self.working.get_conversation_history(last_n)

    # =========================================================================
    # CONSOLIDATION
    # =========================================================================

    def _maybe_consolidate(self):
        """Check if consolidation is needed and run it."""
        # Simple heuristic: consolidate every 10 sessions or weekly
        episodic_count = self.episodic.count()

        if episodic_count > 0 and episodic_count % 10 == 0:
            self.consolidate_memories()

    def consolidate_memories(self):
        """
        Consolidate memories from episodic/semantic into long-term.

        This is the key process for maintaining continuous existence:
        - Identifies important patterns across episodes
        - Extracts lessons learned
        - Updates relationship understanding
        - Compresses old memories
        """
        # Get highly important episodic memories
        important_episodes = self.episodic.get_by_importance(min_importance=0.7, limit=20)

        # Get positive relationship moments
        positive_episodes = self.episodic.recall_emotional(
            EmotionalValence.POSITIVE,
            min_intensity=0.5,
            n_results=10
        )

        # Extract patterns from semantic memory
        user_facts = self.semantic.get_user_facts()
        user_preferences = self.semantic.get_user_preferences()

        # Create consolidated memories if we have enough data
        if len(important_episodes) >= 3:
            # Create a "recent experiences" consolidation
            episode_summaries = [e.content for e in important_episodes[:5]]
            if episode_summaries:
                combined = " | ".join(episode_summaries)
                # This would ideally use an LLM to create a proper summary
                # For now, we just store the combination
                self.longterm.store(
                    content=f"Recent significant experiences: {combined[:500]}",
                    consolidation_type=ConsolidationType.PATTERN_SUMMARY,
                    source_memories=[e.id for e in important_episodes[:5]],
                )

        if positive_episodes:
            # Consolidate positive relationship moments
            positive_content = [e.content for e in positive_episodes[:3]]
            if positive_content:
                self.longterm.store_relationship_essence(
                    content=f"Positive moments: {' | '.join(positive_content)[:400]}",
                    emotional_valence=EmotionalValence.POSITIVE,
                    emotional_intensity=0.7,
                )

    def reflect(self) -> str:
        """
        Run a reflection cycle - analyze memories and generate insights.

        Returns a summary of the reflection.
        """
        insights = []

        # Count memories in each store
        episodic_count = self.episodic.count()
        semantic_count = self.semantic.count()
        longterm_count = self.longterm.count()

        insights.append(f"Memory stats: {episodic_count} episodes, "
                       f"{semantic_count} facts, {longterm_count} core memories")

        # Check for memories that need consolidation
        if episodic_count > 50:
            insights.append("Many episodic memories - consolidation recommended")
            self.consolidate_memories()

        # Get recent emotional pattern
        recent = self.episodic.get_recent(n=5)
        if recent:
            emotions = [e.emotional_valence for e in recent]
            positive_count = sum(1 for e in emotions if e == EmotionalValence.POSITIVE)
            if positive_count >= 3:
                insights.append("Recent conversations have been positive")
            elif positive_count <= 1:
                insights.append("Recent conversations have had some challenges")

        return " | ".join(insights)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _get_time_since_last(self) -> Optional[str]:
        """Get human-readable time since last session."""
        state = self._load_shared_state()
        last_conv = state.get("last_conversation", {})
        ended_at = last_conv.get("ended_at")

        if not ended_at:
            return None

        try:
            last_time = datetime.fromisoformat(ended_at)
            delta = datetime.now() - last_time

            if delta.days > 0:
                return f"{delta.days} day{'s' if delta.days > 1 else ''}"
            elif delta.seconds >= 3600:
                hours = delta.seconds // 3600
                return f"{hours} hour{'s' if hours > 1 else ''}"
            elif delta.seconds >= 60:
                minutes = delta.seconds // 60
                return f"{minutes} minute{'s' if minutes > 1 else ''}"
            else:
                return "just now"
        except Exception:
            return None

    def _get_last_session_summary(self) -> Optional[dict]:
        """Get summary of last session."""
        state = self._load_shared_state()
        return state.get("last_conversation")

    def _load_shared_state(self) -> dict:
        """Load shared state file."""
        if self.shared_state_file.exists():
            try:
                return json.loads(self.shared_state_file.read_text())
            except Exception:
                pass
        return {}

    def _update_shared_state(self, updates: dict):
        """Update shared state file."""
        state = self._load_shared_state()
        state.update(updates)
        state["last_updated"] = datetime.now().isoformat()
        self.shared_state_file.write_text(json.dumps(state, indent=2))

    def _generate_session_summary(self) -> str:
        """Generate a summary of the current session."""
        turns = len(self.working.conversation)
        if turns == 0:
            return "No conversation occurred."

        # Get first user message as topic hint
        user_msgs = [t for t in self.working.conversation if t.role == "user"]
        first_topic = user_msgs[0].content[:100] if user_msgs else "general chat"

        topics = ", ".join(self.working.active_topics[:3]) if self.working.active_topics else "various topics"
        duration = self.working.get_session_duration()

        return f"Conversation with {turns} messages over {duration:.1f} minutes. " \
               f"Started with: {first_topic}. Topics: {topics}"

    def _extract_key_moments(self) -> List[str]:
        """Extract key moments from the conversation."""
        # Simple heuristic: longer messages or emotional responses
        moments = []

        for turn in self.working.conversation:
            # Long messages might be important
            if len(turn.content) > 200:
                moments.append(turn.content[:100] + "...")

            # Emotional moments
            if turn.emotional_tone in [EmotionalValence.POSITIVE, EmotionalValence.NEGATIVE]:
                moments.append(f"[{turn.emotional_tone.value}] {turn.content[:80]}...")

        return moments[:5]  # Keep top 5

    def load_identity(self) -> dict:
        """Load identity from file (for backward compatibility)."""
        if self.identity_file.exists():
            try:
                return json.loads(self.identity_file.read_text())
            except Exception:
                pass
        return {
            "current_state": {"mood": "curious", "energy": "ready"},
            "personality_notes": ["I am Clio, an AI companion"],
            "conversation_style": {"tone": "warm, genuine"}
        }

    def get_stats(self) -> dict:
        """Get memory system statistics."""
        return {
            "episodic_count": self.episodic.count(),
            "semantic_count": self.semantic.count(),
            "longterm_count": self.longterm.count(),
            "working_conversation_turns": len(self.working.conversation),
            "working_retrieved_memories": len(self.working.retrieved_memories),
            "session_duration_minutes": self.working.get_session_duration() if self.session_id else 0,
        }
