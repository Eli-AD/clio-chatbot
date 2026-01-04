"""Episodic Memory - Stores experiences and events with temporal context."""

from datetime import datetime, timedelta
from typing import List, Optional

from .base import BaseMemory, MemoryEntry, MemoryType, EmotionalValence


class EpisodicMemory(BaseMemory):
    """
    Episodic Memory - "Remember when..."

    Stores specific experiences and events with:
    - Precise timestamps
    - Emotional coloring
    - Narrative connections to other episodes
    - Context about what was happening

    Examples:
    - "Remember when we debugged that memory leak together?"
    - "Last Tuesday we talked about your vacation plans"
    - "You seemed frustrated when working on the API"
    """

    def __init__(self):
        super().__init__(collection_name="clio_episodic")

    def store(
        self,
        content: str,
        importance: float = 0.5,
        emotional_valence: EmotionalValence = EmotionalValence.NEUTRAL,
        emotional_intensity: float = 0.0,
        tags: List[str] = None,
        source: str = "conversation",
        context: dict = None,
        related_episodes: List[str] = None,
    ) -> MemoryEntry:
        """
        Store an episodic memory (an experience/event).

        Args:
            content: What happened - the experience description
            importance: How significant this episode is (0.0-1.0)
            emotional_valence: Emotional tone (positive/negative/neutral/mixed)
            emotional_intensity: How emotionally charged (0.0-1.0)
            tags: Topic/category tags
            source: Where this came from (conversation, reflection, observation)
            context: Additional context (time of day, what we were doing, etc.)
            related_episodes: IDs of related episodic memories
        """
        entry = MemoryEntry(
            id=self._generate_id("episode"),
            content=content,
            memory_type=MemoryType.EPISODIC,
            timestamp=datetime.now(),
            importance=importance,
            emotional_valence=emotional_valence,
            emotional_intensity=emotional_intensity,
            tags=tags or [],
            source=source,
            related_memories=related_episodes or [],
            decay_rate=0.05,  # Episodes decay slowly
            metadata=context or {},
        )

        self._store_in_chroma(entry)
        return entry

    def recall(
        self,
        query: str,
        n_results: int = 5,
        time_filter: Optional[str] = None,  # "today", "week", "month", "all"
        min_importance: float = 0.0,
    ) -> List[MemoryEntry]:
        """
        Recall episodic memories relevant to query.

        Args:
            query: What to search for
            n_results: Maximum results to return
            time_filter: Filter by time period
            min_importance: Minimum importance threshold
        """
        # Build where filter
        where_filter = {"memory_type": MemoryType.EPISODIC.value}

        entries = self._recall_from_chroma(
            query=query,
            n_results=n_results * 2,  # Get extra to filter
            where=where_filter
        )

        # Apply time filter
        if time_filter and time_filter != "all":
            now = datetime.now()
            if time_filter == "today":
                cutoff = now - timedelta(days=1)
            elif time_filter == "week":
                cutoff = now - timedelta(weeks=1)
            elif time_filter == "month":
                cutoff = now - timedelta(days=30)
            else:
                cutoff = None

            if cutoff:
                entries = [e for e in entries if e.timestamp >= cutoff]

        # Apply importance filter
        if min_importance > 0:
            entries = [e for e in entries if e.get_effective_importance() >= min_importance]

        # Update access tracking
        for entry in entries[:n_results]:
            self._update_access(entry.id)

        return entries[:n_results]

    def recall_by_time(
        self,
        start: datetime,
        end: datetime = None,
        n_results: int = 10
    ) -> List[MemoryEntry]:
        """Recall episodes from a specific time period."""
        end = end or datetime.now()

        # Get all episodes and filter by time
        # Note: ChromaDB doesn't support datetime range queries well
        all_results = self.collection.get(
            limit=n_results * 5,
            include=["documents", "metadatas"],
            where={"memory_type": MemoryType.EPISODIC.value}
        )

        entries = []
        if all_results and all_results["documents"]:
            for i, doc in enumerate(all_results["documents"]):
                meta = all_results["metadatas"][i] if all_results["metadatas"] else {}
                timestamp = datetime.fromisoformat(meta["timestamp"]) if meta.get("timestamp") else datetime.now()

                if start <= timestamp <= end:
                    entry = MemoryEntry(
                        id=all_results["ids"][i],
                        content=doc,
                        memory_type=MemoryType.EPISODIC,
                        timestamp=timestamp,
                        importance=meta.get("importance", 0.5),
                        emotional_valence=EmotionalValence(meta.get("emotional_valence", "neutral")),
                        emotional_intensity=meta.get("emotional_intensity", 0.0),
                        tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                        source=meta.get("source", "unknown"),
                    )
                    entries.append(entry)

        # Sort by timestamp descending
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:n_results]

    def recall_emotional(
        self,
        valence: EmotionalValence,
        min_intensity: float = 0.3,
        n_results: int = 5
    ) -> List[MemoryEntry]:
        """Recall episodes with specific emotional qualities."""
        results = self.collection.get(
            limit=n_results * 3,
            include=["documents", "metadatas"],
            where={
                "$and": [
                    {"memory_type": MemoryType.EPISODIC.value},
                    {"emotional_valence": valence.value},
                ]
            }
        )

        entries = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                intensity = meta.get("emotional_intensity", 0.0)

                if intensity >= min_intensity:
                    entry = MemoryEntry(
                        id=results["ids"][i],
                        content=doc,
                        memory_type=MemoryType.EPISODIC,
                        timestamp=datetime.fromisoformat(meta["timestamp"]) if meta.get("timestamp") else datetime.now(),
                        importance=meta.get("importance", 0.5),
                        emotional_valence=valence,
                        emotional_intensity=intensity,
                        tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                        source=meta.get("source", "unknown"),
                    )
                    entries.append(entry)

        # Sort by emotional intensity
        entries.sort(key=lambda e: e.emotional_intensity, reverse=True)
        return entries[:n_results]

    def get_recent(self, n: int = 5) -> List[MemoryEntry]:
        """Get most recent episodes."""
        return self.recall_by_time(
            start=datetime.now() - timedelta(days=30),
            n_results=n
        )

    def get_narrative_thread(self, episode_id: str, depth: int = 3) -> List[MemoryEntry]:
        """
        Get a chain of related episodes (narrative thread).

        Follows related_memories links to build a connected story.
        """
        visited = set()
        thread = []

        def follow_thread(mem_id: str, current_depth: int):
            if current_depth <= 0 or mem_id in visited:
                return

            visited.add(mem_id)

            result = self.collection.get(ids=[mem_id], include=["documents", "metadatas"])
            if result and result["documents"]:
                doc = result["documents"][0]
                meta = result["metadatas"][0] if result["metadatas"] else {}

                entry = MemoryEntry(
                    id=mem_id,
                    content=doc,
                    memory_type=MemoryType.EPISODIC,
                    timestamp=datetime.fromisoformat(meta["timestamp"]) if meta.get("timestamp") else datetime.now(),
                    importance=meta.get("importance", 0.5),
                    emotional_valence=EmotionalValence(meta.get("emotional_valence", "neutral")),
                    emotional_intensity=meta.get("emotional_intensity", 0.0),
                    tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                    source=meta.get("source", "unknown"),
                )
                thread.append(entry)

                # Follow related memories
                related = meta.get("related_memories", "").split(",")
                for rel_id in related:
                    if rel_id.strip():
                        follow_thread(rel_id.strip(), current_depth - 1)

        follow_thread(episode_id, depth)

        # Sort by timestamp
        thread.sort(key=lambda e: e.timestamp)
        return thread

    def store_conversation_episode(
        self,
        summary: str,
        topics: List[str],
        emotional_valence: EmotionalValence,
        emotional_intensity: float,
        key_moments: List[str] = None,
        duration_minutes: float = 0,
    ) -> MemoryEntry:
        """
        Store a complete conversation as an episode.

        Called at the end of a session to create an episodic memory
        of the entire conversation.
        """
        context = {
            "type": "conversation",
            "topics": topics,
            "duration_minutes": duration_minutes,
            "key_moments": key_moments or [],
            "time_of_day": datetime.now().strftime("%H:%M"),
            "day_of_week": datetime.now().strftime("%A"),
        }

        # Importance based on length and emotional intensity
        importance = min(1.0, 0.4 + (emotional_intensity * 0.3) + (duration_minutes / 60 * 0.3))

        return self.store(
            content=summary,
            importance=importance,
            emotional_valence=emotional_valence,
            emotional_intensity=emotional_intensity,
            tags=topics,
            source="conversation",
            context=context,
        )
