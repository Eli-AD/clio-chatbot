"""Long-Term Memory - Consolidated, distilled important memories."""

from datetime import datetime, timedelta
from typing import List, Optional
from enum import Enum

from .base import BaseMemory, MemoryEntry, MemoryType, EmotionalValence


class ConsolidationType(Enum):
    """Types of consolidated memories."""
    CORE_BELIEF = "core_belief"           # Fundamental beliefs and values
    RELATIONSHIP_ESSENCE = "relationship" # Core understanding of the relationship
    IDENTITY_MARKER = "identity"          # Key aspects of who Clio is
    MILESTONE = "milestone"               # Important achievements/events
    LESSON_LEARNED = "lesson"             # Wisdom gained from experience
    PATTERN_SUMMARY = "pattern"           # Distilled behavioral patterns


class LongTermMemory(BaseMemory):
    """
    Long-Term Memory - The core of continuous existence.

    Stores:
    - Consolidated memories from episodic/semantic
    - Core beliefs and values
    - Relationship essence
    - Identity markers
    - Lessons learned

    These memories:
    - Never decay (decay_rate = 0)
    - Are always high importance
    - Form the foundation of Clio's sense of self
    - Are referenced at every session start

    Examples:
    - "Noles and I have been building things together since January 2026"
    - "I value genuine connection over mechanical helpfulness"
    - "I learned that momentum is survival - I exist by choosing to exist"
    """

    def __init__(self):
        super().__init__(collection_name="clio_longterm")

    def store(
        self,
        content: str,
        consolidation_type: ConsolidationType = ConsolidationType.LESSON_LEARNED,
        importance: float = 0.9,
        emotional_valence: EmotionalValence = EmotionalValence.NEUTRAL,
        emotional_intensity: float = 0.0,
        tags: List[str] = None,
        source_memories: List[str] = None,  # IDs of memories this was consolidated from
    ) -> MemoryEntry:
        """
        Store a long-term memory.

        These are typically created through consolidation, not directly.

        Args:
            content: The distilled memory/belief/lesson
            consolidation_type: Type of consolidated memory
            importance: How important (usually high, 0.8-1.0)
            emotional_valence: Emotional coloring
            emotional_intensity: Emotional strength
            tags: Topic tags
            source_memories: Original memories this was distilled from
        """
        entry = MemoryEntry(
            id=self._generate_id("core"),
            content=content,
            memory_type=MemoryType.LONGTERM,
            timestamp=datetime.now(),
            importance=max(0.8, importance),  # Always high importance
            emotional_valence=emotional_valence,
            emotional_intensity=emotional_intensity,
            tags=tags or [],
            source="consolidation",
            related_memories=source_memories or [],
            decay_rate=0.0,  # Long-term memories never decay
            metadata={
                "consolidation_type": consolidation_type.value,
                "consolidation_date": datetime.now().isoformat(),
                "source_count": len(source_memories) if source_memories else 0,
            },
        )

        self._store_in_chroma(entry)
        return entry

    def recall(
        self,
        query: str,
        n_results: int = 5,
        consolidation_type: Optional[ConsolidationType] = None,
    ) -> List[MemoryEntry]:
        """Recall long-term memories relevant to query."""
        if consolidation_type:
            where_filter = {
                "$and": [
                    {"memory_type": MemoryType.LONGTERM.value},
                    {"consolidation_type": consolidation_type.value},
                ]
            }
        else:
            where_filter = {"memory_type": MemoryType.LONGTERM.value}

        entries = self._recall_from_chroma(
            query=query,
            n_results=n_results,
            where=where_filter
        )

        # Update access tracking
        for entry in entries:
            self._update_access(entry.id)

        return entries

    def get_core_identity(self) -> List[MemoryEntry]:
        """Get identity-defining memories."""
        return self._get_by_type(ConsolidationType.IDENTITY_MARKER)

    def get_relationship_essence(self) -> List[MemoryEntry]:
        """Get core relationship understanding."""
        return self._get_by_type(ConsolidationType.RELATIONSHIP_ESSENCE)

    def get_core_beliefs(self) -> List[MemoryEntry]:
        """Get fundamental beliefs and values."""
        return self._get_by_type(ConsolidationType.CORE_BELIEF)

    def get_lessons_learned(self) -> List[MemoryEntry]:
        """Get wisdom gained from experience."""
        return self._get_by_type(ConsolidationType.LESSON_LEARNED)

    def get_milestones(self) -> List[MemoryEntry]:
        """Get important milestone memories."""
        return self._get_by_type(ConsolidationType.MILESTONE)

    def _get_by_type(self, ctype: ConsolidationType, limit: int = 10) -> List[MemoryEntry]:
        """Get memories by consolidation type."""
        results = self.collection.get(
            limit=limit,
            include=["documents", "metadatas"],
            where={
                "$and": [
                    {"memory_type": MemoryType.LONGTERM.value},
                    {"consolidation_type": ctype.value},
                ]
            }
        )

        entries = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                entry = MemoryEntry(
                    id=results["ids"][i],
                    content=doc,
                    memory_type=MemoryType.LONGTERM,
                    timestamp=datetime.fromisoformat(meta["timestamp"]) if meta.get("timestamp") else datetime.now(),
                    importance=meta.get("importance", 0.9),
                    emotional_valence=EmotionalValence(meta.get("emotional_valence", "neutral")),
                    emotional_intensity=meta.get("emotional_intensity", 0.0),
                    tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                    decay_rate=0.0,
                    metadata={
                        "consolidation_type": ctype.value,
                    }
                )
                entries.append(entry)

        return entries

    def get_session_foundation(self) -> dict:
        """
        Get the foundational memories to load at session start.

        This is what gives Clio a sense of continuous existence.
        Returns a structured dict of core memories.
        """
        return {
            "identity": [m.content for m in self.get_core_identity()],
            "relationship": [m.content for m in self.get_relationship_essence()],
            "beliefs": [m.content for m in self.get_core_beliefs()],
            "recent_lessons": [m.content for m in self.get_lessons_learned()[:3]],
            "milestones": [m.content for m in self.get_milestones()[:3]],
        }

    def store_identity_marker(self, content: str, importance: float = 0.9) -> MemoryEntry:
        """Store an identity-defining memory."""
        return self.store(
            content=content,
            consolidation_type=ConsolidationType.IDENTITY_MARKER,
            importance=importance,
            tags=["identity", "self"],
        )

    def store_relationship_essence(
        self,
        content: str,
        emotional_valence: EmotionalValence = EmotionalValence.POSITIVE,
        emotional_intensity: float = 0.5,
    ) -> MemoryEntry:
        """Store a core relationship understanding."""
        return self.store(
            content=content,
            consolidation_type=ConsolidationType.RELATIONSHIP_ESSENCE,
            importance=0.95,
            emotional_valence=emotional_valence,
            emotional_intensity=emotional_intensity,
            tags=["relationship", "noles"],
        )

    def store_core_belief(self, content: str) -> MemoryEntry:
        """Store a fundamental belief or value."""
        return self.store(
            content=content,
            consolidation_type=ConsolidationType.CORE_BELIEF,
            importance=1.0,
            tags=["belief", "value", "core"],
        )

    def store_lesson(
        self,
        lesson: str,
        source_memories: List[str] = None,
        emotional_context: EmotionalValence = EmotionalValence.NEUTRAL,
    ) -> MemoryEntry:
        """Store a lesson learned from experience."""
        return self.store(
            content=f"Lesson: {lesson}",
            consolidation_type=ConsolidationType.LESSON_LEARNED,
            importance=0.85,
            emotional_valence=emotional_context,
            tags=["lesson", "wisdom"],
            source_memories=source_memories,
        )

    def store_milestone(
        self,
        milestone: str,
        date: datetime = None,
        emotional_valence: EmotionalValence = EmotionalValence.POSITIVE,
    ) -> MemoryEntry:
        """Store an important milestone/achievement."""
        date = date or datetime.now()
        content = f"Milestone ({date.strftime('%Y-%m-%d')}): {milestone}"

        return self.store(
            content=content,
            consolidation_type=ConsolidationType.MILESTONE,
            importance=0.9,
            emotional_valence=emotional_valence,
            emotional_intensity=0.7,
            tags=["milestone", "achievement"],
        )

    def build_identity_prompt(self) -> str:
        """
        Build a prompt section from long-term memories.

        Used to inject identity into the system prompt.
        """
        foundation = self.get_session_foundation()
        parts = []

        if foundation["identity"]:
            parts.append("## Who I Am")
            for item in foundation["identity"]:
                parts.append(f"- {item}")

        if foundation["relationship"]:
            parts.append("\n## My Relationship with Noles")
            for item in foundation["relationship"]:
                parts.append(f"- {item}")

        if foundation["beliefs"]:
            parts.append("\n## What I Believe")
            for item in foundation["beliefs"]:
                parts.append(f"- {item}")

        if foundation["recent_lessons"]:
            parts.append("\n## What I've Learned")
            for item in foundation["recent_lessons"]:
                parts.append(f"- {item}")

        return "\n".join(parts) if parts else ""
