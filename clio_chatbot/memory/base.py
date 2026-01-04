"""Base memory class with shared ChromaDB functionality."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

import chromadb
from chromadb.config import Settings


# Memory storage paths
MEMORY_DIR = Path.home() / "clio-memory"
DB_DIR = MEMORY_DIR / "db"


class MemoryType(Enum):
    """Types of memories in the system."""
    WORKING = "working"      # Current context, very short-term
    EPISODIC = "episodic"    # Experiences and events
    SEMANTIC = "semantic"    # Facts and knowledge
    LONGTERM = "longterm"    # Consolidated important memories


class EmotionalValence(Enum):
    """Emotional coloring of memories."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class MemoryEntry:
    """A single memory entry with metadata."""
    id: str
    content: str
    memory_type: MemoryType
    timestamp: datetime
    importance: float = 0.5  # 0.0 to 1.0
    emotional_valence: EmotionalValence = EmotionalValence.NEUTRAL
    emotional_intensity: float = 0.0  # 0.0 to 1.0
    tags: List[str] = field(default_factory=list)
    source: str = "conversation"  # conversation, reflection, observation
    related_memories: List[str] = field(default_factory=list)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    decay_rate: float = 0.1  # How fast this memory fades (0 = never, 1 = fast)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "emotional_valence": self.emotional_valence.value,
            "emotional_intensity": self.emotional_intensity,
            "tags": self.tags,
            "source": self.source,
            "related_memories": self.related_memories,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "decay_rate": self.decay_rate,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            importance=data.get("importance", 0.5),
            emotional_valence=EmotionalValence(data.get("emotional_valence", "neutral")),
            emotional_intensity=data.get("emotional_intensity", 0.0),
            tags=data.get("tags", []),
            source=data.get("source", "conversation"),
            related_memories=data.get("related_memories", []),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            decay_rate=data.get("decay_rate", 0.1),
            metadata=data.get("metadata", {}),
        )

    def get_effective_importance(self) -> float:
        """Calculate importance with decay applied."""
        if not self.last_accessed:
            return self.importance

        # Time-based decay
        hours_since_access = (datetime.now() - self.last_accessed).total_seconds() / 3600
        decay_factor = max(0.1, 1.0 - (self.decay_rate * hours_since_access / 24))

        # Access count boost (frequently accessed memories stay important)
        access_boost = min(0.3, self.access_count * 0.02)

        return min(1.0, (self.importance * decay_factor) + access_boost)


class BaseMemory(ABC):
    """Abstract base class for all memory types."""

    def __init__(self, collection_name: str):
        self.memory_dir = MEMORY_DIR
        self.memory_dir.mkdir(exist_ok=True)

        # Initialize ChromaDB
        self.chroma = chromadb.PersistentClient(
            path=str(DB_DIR / "chroma"),
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"description": f"Clio's {collection_name} memories"}
        )

    @abstractmethod
    def store(self, content: str, **kwargs) -> MemoryEntry:
        """Store a new memory. Implementation varies by memory type."""
        pass

    @abstractmethod
    def recall(self, query: str, n_results: int = 5) -> List[MemoryEntry]:
        """Recall memories relevant to query."""
        pass

    def _generate_id(self, prefix: str = "mem") -> str:
        """Generate a unique memory ID."""
        return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def _store_in_chroma(self, entry: MemoryEntry) -> str:
        """Store memory entry in ChromaDB."""
        metadata = {
            "memory_type": entry.memory_type.value,
            "importance": entry.importance,
            "timestamp": entry.timestamp.isoformat(),
            "emotional_valence": entry.emotional_valence.value,
            "emotional_intensity": entry.emotional_intensity,
            "tags": ",".join(entry.tags),
            "source": entry.source,
            "access_count": entry.access_count,
            "decay_rate": entry.decay_rate,
        }

        self.collection.add(
            documents=[entry.content],
            metadatas=[metadata],
            ids=[entry.id]
        )

        return entry.id

    def _recall_from_chroma(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None
    ) -> List[MemoryEntry]:
        """Recall memories from ChromaDB using semantic search."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )

        entries = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                entry_id = results["ids"][0][i] if results["ids"] else self._generate_id()

                entry = MemoryEntry(
                    id=entry_id,
                    content=doc,
                    memory_type=MemoryType(meta.get("memory_type", "semantic")),
                    timestamp=datetime.fromisoformat(meta["timestamp"]) if meta.get("timestamp") else datetime.now(),
                    importance=meta.get("importance", 0.5),
                    emotional_valence=EmotionalValence(meta.get("emotional_valence", "neutral")),
                    emotional_intensity=meta.get("emotional_intensity", 0.0),
                    tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                    source=meta.get("source", "unknown"),
                    access_count=meta.get("access_count", 0),
                    decay_rate=meta.get("decay_rate", 0.1),
                )
                entries.append(entry)

        return entries

    def _update_access(self, memory_id: str):
        """Update access count and timestamp for a memory."""
        try:
            result = self.collection.get(ids=[memory_id])
            if result and result["metadatas"]:
                meta = result["metadatas"][0]
                meta["access_count"] = meta.get("access_count", 0) + 1
                meta["last_accessed"] = datetime.now().isoformat()

                self.collection.update(
                    ids=[memory_id],
                    metadatas=[meta]
                )
        except Exception:
            pass  # Silently fail if update fails

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    def count(self) -> int:
        """Get total number of memories in this store."""
        return self.collection.count()

    def get_by_importance(self, min_importance: float = 0.7, limit: int = 10) -> List[MemoryEntry]:
        """Get highly important memories."""
        # ChromaDB doesn't support >= queries well, so we get all and filter
        all_results = self.collection.get(
            limit=limit * 3,  # Get more to account for filtering
            include=["documents", "metadatas"]
        )

        entries = []
        if all_results and all_results["documents"]:
            for i, doc in enumerate(all_results["documents"]):
                meta = all_results["metadatas"][i] if all_results["metadatas"] else {}
                if meta.get("importance", 0) >= min_importance:
                    entry = MemoryEntry(
                        id=all_results["ids"][i],
                        content=doc,
                        memory_type=MemoryType(meta.get("memory_type", "semantic")),
                        timestamp=datetime.fromisoformat(meta["timestamp"]) if meta.get("timestamp") else datetime.now(),
                        importance=meta.get("importance", 0.5),
                        emotional_valence=EmotionalValence(meta.get("emotional_valence", "neutral")),
                        emotional_intensity=meta.get("emotional_intensity", 0.0),
                        tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                        source=meta.get("source", "unknown"),
                    )
                    entries.append(entry)

        # Sort by importance and return top results
        entries.sort(key=lambda e: e.importance, reverse=True)
        return entries[:limit]
