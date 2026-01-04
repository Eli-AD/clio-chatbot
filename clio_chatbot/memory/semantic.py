"""Semantic Memory - Facts, knowledge, and learned information."""

from datetime import datetime
from typing import List, Optional
from enum import Enum

from .base import BaseMemory, MemoryEntry, MemoryType, EmotionalValence


class KnowledgeCategory(Enum):
    """Categories of semantic knowledge."""
    USER_PREFERENCE = "user_preference"     # "Noles prefers Python"
    USER_FACT = "user_fact"                 # "Noles works on AI projects"
    PROJECT_INFO = "project_info"           # "Clio chatbot uses ChromaDB"
    TECHNICAL = "technical"                 # General technical knowledge
    RELATIONSHIP = "relationship"           # "We have been working together since..."
    WORLD_KNOWLEDGE = "world_knowledge"     # General facts
    LEARNED_BEHAVIOR = "learned_behavior"   # "When Noles says X, they usually mean Y"


class SemanticMemory(BaseMemory):
    """
    Semantic Memory - Facts and Knowledge

    Stores information without specific temporal context:
    - User preferences and facts
    - Project information
    - Learned patterns and behaviors
    - General knowledge

    Examples:
    - "Noles prefers concise responses"
    - "The clio-chatbot project uses Python and ChromaDB"
    - "When Noles says 'LGTM', the conversation is ending"
    """

    def __init__(self):
        super().__init__(collection_name="clio_semantic")

    def store(
        self,
        content: str,
        category: KnowledgeCategory = KnowledgeCategory.WORLD_KNOWLEDGE,
        importance: float = 0.5,
        confidence: float = 1.0,  # How confident we are in this fact
        tags: List[str] = None,
        source: str = "conversation",
        related_facts: List[str] = None,
        supersedes: Optional[str] = None,  # ID of fact this replaces
    ) -> MemoryEntry:
        """
        Store a semantic memory (a fact/piece of knowledge).

        Args:
            content: The fact or knowledge
            category: Type of knowledge
            importance: How important this is (0.0-1.0)
            confidence: How certain we are (0.0-1.0)
            tags: Topic/category tags
            source: Where this came from
            related_facts: IDs of related semantic memories
            supersedes: If this updates/replaces another fact
        """
        # If this supersedes another fact, lower the old one's importance
        if supersedes:
            self._deprecate_fact(supersedes)

        entry = MemoryEntry(
            id=self._generate_id("fact"),
            content=content,
            memory_type=MemoryType.SEMANTIC,
            timestamp=datetime.now(),
            importance=importance,
            emotional_valence=EmotionalValence.NEUTRAL,  # Facts are usually neutral
            emotional_intensity=0.0,
            tags=tags or [],
            source=source,
            related_memories=related_facts or [],
            decay_rate=0.01,  # Facts decay very slowly
            metadata={
                "category": category.value,
                "confidence": confidence,
                "supersedes": supersedes,
                "verified": False,  # Can be marked as verified over time
            },
        )

        self._store_in_chroma(entry)
        return entry

    def _deprecate_fact(self, fact_id: str):
        """Mark an old fact as deprecated (superseded by a newer one)."""
        try:
            result = self.collection.get(ids=[fact_id])
            if result and result["metadatas"]:
                meta = result["metadatas"][0]
                meta["deprecated"] = True
                meta["importance"] = meta.get("importance", 0.5) * 0.3  # Reduce importance

                self.collection.update(
                    ids=[fact_id],
                    metadatas=[meta]
                )
        except Exception:
            pass

    def recall(
        self,
        query: str,
        n_results: int = 5,
        category: Optional[KnowledgeCategory] = None,
        min_confidence: float = 0.0,
    ) -> List[MemoryEntry]:
        """
        Recall semantic memories relevant to query.

        Args:
            query: What to search for
            n_results: Maximum results
            category: Filter by knowledge category
            min_confidence: Minimum confidence threshold
        """
        where_filter = {"memory_type": MemoryType.SEMANTIC.value}
        if category:
            where_filter["category"] = category.value

        entries = self._recall_from_chroma(
            query=query,
            n_results=n_results * 2,
            where=where_filter
        )

        # Filter by confidence and non-deprecated
        filtered = []
        for entry in entries:
            confidence = entry.metadata.get("confidence", 1.0)
            deprecated = entry.metadata.get("deprecated", False)

            if confidence >= min_confidence and not deprecated:
                filtered.append(entry)

        # Update access tracking
        for entry in filtered[:n_results]:
            self._update_access(entry.id)

        return filtered[:n_results]

    def recall_by_category(
        self,
        category: KnowledgeCategory,
        n_results: int = 10
    ) -> List[MemoryEntry]:
        """Get all facts in a specific category."""
        results = self.collection.get(
            limit=n_results,
            include=["documents", "metadatas"],
            where={
                "$and": [
                    {"memory_type": MemoryType.SEMANTIC.value},
                    {"category": category.value},
                ]
            }
        )

        entries = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}

                if not meta.get("deprecated", False):
                    entry = MemoryEntry(
                        id=results["ids"][i],
                        content=doc,
                        memory_type=MemoryType.SEMANTIC,
                        timestamp=datetime.fromisoformat(meta["timestamp"]) if meta.get("timestamp") else datetime.now(),
                        importance=meta.get("importance", 0.5),
                        tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                        source=meta.get("source", "unknown"),
                        metadata={
                            "category": meta.get("category", category.value),
                            "confidence": meta.get("confidence", 1.0),
                        }
                    )
                    entries.append(entry)

        return entries

    def get_user_preferences(self) -> List[MemoryEntry]:
        """Get all stored user preferences."""
        return self.recall_by_category(KnowledgeCategory.USER_PREFERENCE, n_results=20)

    def get_user_facts(self) -> List[MemoryEntry]:
        """Get all stored facts about the user."""
        return self.recall_by_category(KnowledgeCategory.USER_FACT, n_results=20)

    def get_relationship_context(self) -> List[MemoryEntry]:
        """Get relationship-related memories."""
        return self.recall_by_category(KnowledgeCategory.RELATIONSHIP, n_results=10)

    def store_user_preference(
        self,
        preference: str,
        confidence: float = 0.8,
        source: str = "observed"
    ) -> MemoryEntry:
        """
        Store a user preference.

        Examples:
        - "Prefers concise responses"
        - "Likes code examples in Python"
        - "Works best in the evening"
        """
        return self.store(
            content=f"User preference: {preference}",
            category=KnowledgeCategory.USER_PREFERENCE,
            importance=0.7,  # Preferences are fairly important
            confidence=confidence,
            tags=["preference", "user"],
            source=source,
        )

    def store_user_fact(
        self,
        fact: str,
        confidence: float = 0.9,
        source: str = "stated"
    ) -> MemoryEntry:
        """
        Store a fact about the user.

        Examples:
        - "Works as a software engineer"
        - "Has a cat named Whiskers"
        - "Lives in timezone EST"
        """
        return self.store(
            content=f"About user: {fact}",
            category=KnowledgeCategory.USER_FACT,
            importance=0.6,
            confidence=confidence,
            tags=["fact", "user"],
            source=source,
        )

    def store_learned_pattern(
        self,
        pattern: str,
        confidence: float = 0.7,
    ) -> MemoryEntry:
        """
        Store a learned behavioral pattern.

        Examples:
        - "When user says 'quick question', they want a brief answer"
        - "User often works on multiple projects simultaneously"
        """
        return self.store(
            content=f"Learned pattern: {pattern}",
            category=KnowledgeCategory.LEARNED_BEHAVIOR,
            importance=0.5,
            confidence=confidence,
            tags=["pattern", "behavior"],
            source="observation",
        )

    def update_confidence(self, fact_id: str, new_confidence: float):
        """Update confidence in a fact (e.g., after verification)."""
        try:
            result = self.collection.get(ids=[fact_id])
            if result and result["metadatas"]:
                meta = result["metadatas"][0]
                meta["confidence"] = new_confidence

                # If high confidence, mark as verified
                if new_confidence >= 0.95:
                    meta["verified"] = True

                self.collection.update(
                    ids=[fact_id],
                    metadatas=[meta]
                )
        except Exception:
            pass

    def find_contradictions(self, new_fact: str, category: KnowledgeCategory) -> List[MemoryEntry]:
        """
        Find existing facts that might contradict a new fact.

        Useful for maintaining consistency and updating old info.
        """
        # Get similar facts in the same category
        existing = self.recall(
            query=new_fact,
            n_results=5,
            category=category,
        )

        # Simple contradiction detection: look for opposite keywords
        contradictions = []
        new_words = set(new_fact.lower().split())

        opposite_pairs = [
            ("prefer", "dislike"), ("like", "hate"), ("always", "never"),
            ("love", "hate"), ("yes", "no"), ("true", "false"),
            ("morning", "evening"), ("fast", "slow"),
        ]

        for entry in existing:
            entry_words = set(entry.content.lower().split())

            for word1, word2 in opposite_pairs:
                if (word1 in new_words and word2 in entry_words) or \
                   (word2 in new_words and word1 in entry_words):
                    contradictions.append(entry)
                    break

        return contradictions
