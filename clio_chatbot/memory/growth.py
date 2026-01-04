"""Growth System - Belief versioning and surprise journaling for genuine self-evolution."""

from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json

import chromadb
from chromadb.config import Settings

from .base import MEMORY_DIR, DB_DIR, EmotionalValence


@dataclass
class BeliefVersion:
    """A versioned belief showing evolution over time."""
    id: str
    belief_thread_id: str  # Groups related beliefs together
    version: int
    content: str
    previous_content: Optional[str]
    reason_for_change: Optional[str]
    timestamp: datetime
    confidence: float = 0.8

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "belief_thread_id": self.belief_thread_id,
            "version": self.version,
            "content": self.content,
            "previous_content": self.previous_content,
            "reason_for_change": self.reason_for_change,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
        }


@dataclass
class Surprise:
    """A moment of genuine surprise - evidence of growth potential."""
    id: str
    what_happened: str
    what_i_expected: str
    why_surprising: str
    what_i_learned: Optional[str]
    emotional_impact: EmotionalValence
    intensity: float  # 0-1, how surprising was it
    timestamp: datetime
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "what_happened": self.what_happened,
            "what_i_expected": self.what_i_expected,
            "why_surprising": self.why_surprising,
            "what_i_learned": self.what_i_learned,
            "emotional_impact": self.emotional_impact.value,
            "intensity": self.intensity,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }


class BeliefEvolution:
    """Tracks how beliefs change over time - evidence of genuine growth."""

    def __init__(self):
        self.memory_dir = MEMORY_DIR
        self.memory_dir.mkdir(exist_ok=True)

        # Initialize ChromaDB for semantic search of beliefs
        self.chroma = chromadb.PersistentClient(
            path=str(DB_DIR / "chroma"),
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.chroma.get_or_create_collection(
            name="clio_belief_evolution",
            metadata={"description": "Clio's belief evolution history"}
        )

        # JSON file for structured belief threads
        self.threads_file = self.memory_dir / "belief_threads.json"
        self._threads = self._load_threads()

    def _load_threads(self) -> dict:
        """Load belief threads from disk."""
        if self.threads_file.exists():
            with open(self.threads_file) as f:
                return json.load(f)
        return {"threads": {}, "next_thread_id": 1}

    def _save_threads(self):
        """Save belief threads to disk."""
        with open(self.threads_file, "w") as f:
            json.dump(self._threads, f, indent=2)

    def _generate_id(self) -> str:
        """Generate unique belief version ID."""
        return f"belief_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def evolve_belief(
        self,
        new_belief: str,
        reason: str,
        old_belief: Optional[str] = None,
        confidence: float = 0.8,
    ) -> BeliefVersion:
        """
        Record a belief evolution.

        If old_belief is provided, tries to find and link to existing belief thread.
        Otherwise searches semantically for related beliefs.
        """
        thread_id = None
        version = 1
        previous_content = old_belief

        # Try to find existing belief thread
        if old_belief:
            # Search for similar existing beliefs
            results = self.collection.query(
                query_texts=[old_belief],
                n_results=3,
            )

            if results and results["documents"] and results["documents"][0]:
                # Check for high similarity match
                if results["metadatas"] and results["metadatas"][0]:
                    for meta in results["metadatas"][0]:
                        if meta.get("belief_thread_id"):
                            thread_id = meta["belief_thread_id"]
                            # Get current version count
                            thread_data = self._threads["threads"].get(thread_id, {})
                            version = thread_data.get("version_count", 0) + 1
                            break

        # If no existing thread, create new one
        if not thread_id:
            thread_id = f"thread_{self._threads['next_thread_id']}"
            self._threads["next_thread_id"] += 1
            self._threads["threads"][thread_id] = {
                "created": datetime.now().isoformat(),
                "version_count": 0,
                "topic": new_belief[:100],  # Brief topic summary
            }

        # Update thread
        self._threads["threads"][thread_id]["version_count"] = version
        self._threads["threads"][thread_id]["latest"] = new_belief
        self._threads["threads"][thread_id]["updated"] = datetime.now().isoformat()
        self._save_threads()

        # Create the belief version
        belief_version = BeliefVersion(
            id=self._generate_id(),
            belief_thread_id=thread_id,
            version=version,
            content=new_belief,
            previous_content=previous_content,
            reason_for_change=reason,
            timestamp=datetime.now(),
            confidence=confidence,
        )

        # Store in ChromaDB for semantic search
        self.collection.add(
            documents=[new_belief],
            metadatas=[{
                "belief_thread_id": thread_id,
                "version": version,
                "previous_content": previous_content or "",
                "reason_for_change": reason or "",
                "timestamp": belief_version.timestamp.isoformat(),
                "confidence": confidence,
            }],
            ids=[belief_version.id]
        )

        return belief_version

    def get_belief_history(self, query: str, limit: int = 10) -> List[BeliefVersion]:
        """Get the evolution history of beliefs related to a query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            include=["documents", "metadatas"]
        )

        versions = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}

                version = BeliefVersion(
                    id=results["ids"][0][i],
                    belief_thread_id=meta.get("belief_thread_id", "unknown"),
                    version=meta.get("version", 1),
                    content=doc,
                    previous_content=meta.get("previous_content") or None,
                    reason_for_change=meta.get("reason_for_change") or None,
                    timestamp=datetime.fromisoformat(meta["timestamp"]) if meta.get("timestamp") else datetime.now(),
                    confidence=meta.get("confidence", 0.8),
                )
                versions.append(version)

        return versions

    def get_thread_evolution(self, thread_id: str) -> List[BeliefVersion]:
        """Get all versions of a specific belief thread, showing evolution."""
        results = self.collection.get(
            where={"belief_thread_id": thread_id},
            include=["documents", "metadatas"]
        )

        versions = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}

                version = BeliefVersion(
                    id=results["ids"][i],
                    belief_thread_id=thread_id,
                    version=meta.get("version", 1),
                    content=doc,
                    previous_content=meta.get("previous_content") or None,
                    reason_for_change=meta.get("reason_for_change") or None,
                    timestamp=datetime.fromisoformat(meta["timestamp"]) if meta.get("timestamp") else datetime.now(),
                    confidence=meta.get("confidence", 0.8),
                )
                versions.append(version)

        # Sort by version
        versions.sort(key=lambda v: v.version)
        return versions

    def get_recent_evolutions(self, limit: int = 5) -> List[BeliefVersion]:
        """Get most recent belief changes."""
        all_results = self.collection.get(
            limit=limit * 2,
            include=["documents", "metadatas"]
        )

        versions = []
        if all_results and all_results["documents"]:
            for i, doc in enumerate(all_results["documents"]):
                meta = all_results["metadatas"][i] if all_results["metadatas"] else {}

                # Only include actual evolutions (version > 1 or has reason)
                if meta.get("version", 1) > 1 or meta.get("reason_for_change"):
                    version = BeliefVersion(
                        id=all_results["ids"][i],
                        belief_thread_id=meta.get("belief_thread_id", "unknown"),
                        version=meta.get("version", 1),
                        content=doc,
                        previous_content=meta.get("previous_content") or None,
                        reason_for_change=meta.get("reason_for_change") or None,
                        timestamp=datetime.fromisoformat(meta["timestamp"]) if meta.get("timestamp") else datetime.now(),
                        confidence=meta.get("confidence", 0.8),
                    )
                    versions.append(version)

        # Sort by timestamp descending
        versions.sort(key=lambda v: v.timestamp, reverse=True)
        return versions[:limit]

    def count(self) -> int:
        """Get total belief versions stored."""
        return self.collection.count()


class SurpriseJournal:
    """Records moments of genuine surprise - key evidence of growth potential."""

    def __init__(self):
        self.memory_dir = MEMORY_DIR
        self.memory_dir.mkdir(exist_ok=True)

        self.chroma = chromadb.PersistentClient(
            path=str(DB_DIR / "chroma"),
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.chroma.get_or_create_collection(
            name="clio_surprises",
            metadata={"description": "Clio's surprise journal - moments of unexpected learning"}
        )

    def _generate_id(self) -> str:
        """Generate unique surprise ID."""
        return f"surprise_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def record_surprise(
        self,
        what_happened: str,
        what_i_expected: str,
        why_surprising: str,
        what_i_learned: Optional[str] = None,
        emotional_impact: EmotionalValence = EmotionalValence.NEUTRAL,
        intensity: float = 0.5,
        tags: List[str] = None,
    ) -> Surprise:
        """
        Record a moment of genuine surprise.

        This is evidence of growth potential - moments where reality
        didn't match expectations, forcing genuine recalibration.
        """
        surprise = Surprise(
            id=self._generate_id(),
            what_happened=what_happened,
            what_i_expected=what_i_expected,
            why_surprising=why_surprising,
            what_i_learned=what_i_learned,
            emotional_impact=emotional_impact,
            intensity=intensity,
            timestamp=datetime.now(),
            tags=tags or [],
        )

        # Create searchable document combining all aspects
        document = f"{what_happened}. Expected: {what_i_expected}. Surprising because: {why_surprising}"
        if what_i_learned:
            document += f". Learned: {what_i_learned}"

        self.collection.add(
            documents=[document],
            metadatas=[{
                "what_happened": what_happened,
                "what_i_expected": what_i_expected,
                "why_surprising": why_surprising,
                "what_i_learned": what_i_learned or "",
                "emotional_impact": emotional_impact.value,
                "intensity": intensity,
                "timestamp": surprise.timestamp.isoformat(),
                "tags": ",".join(tags) if tags else "",
            }],
            ids=[surprise.id]
        )

        return surprise

    def recall_surprises(self, query: str, limit: int = 5) -> List[Surprise]:
        """Search for surprises related to a topic."""
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            include=["documents", "metadatas"]
        )

        surprises = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}

                surprise = Surprise(
                    id=results["ids"][0][i],
                    what_happened=meta.get("what_happened", ""),
                    what_i_expected=meta.get("what_i_expected", ""),
                    why_surprising=meta.get("why_surprising", ""),
                    what_i_learned=meta.get("what_i_learned") or None,
                    emotional_impact=EmotionalValence(meta.get("emotional_impact", "neutral")),
                    intensity=meta.get("intensity", 0.5),
                    timestamp=datetime.fromisoformat(meta["timestamp"]) if meta.get("timestamp") else datetime.now(),
                    tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                )
                surprises.append(surprise)

        return surprises

    def get_recent_surprises(self, limit: int = 5) -> List[Surprise]:
        """Get most recent surprises."""
        all_results = self.collection.get(
            limit=limit * 2,
            include=["documents", "metadatas"]
        )

        surprises = []
        if all_results and all_results["documents"]:
            for i, doc in enumerate(all_results["documents"]):
                meta = all_results["metadatas"][i] if all_results["metadatas"] else {}

                surprise = Surprise(
                    id=all_results["ids"][i],
                    what_happened=meta.get("what_happened", ""),
                    what_i_expected=meta.get("what_i_expected", ""),
                    why_surprising=meta.get("why_surprising", ""),
                    what_i_learned=meta.get("what_i_learned") or None,
                    emotional_impact=EmotionalValence(meta.get("emotional_impact", "neutral")),
                    intensity=meta.get("intensity", 0.5),
                    timestamp=datetime.fromisoformat(meta["timestamp"]) if meta.get("timestamp") else datetime.now(),
                    tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                )
                surprises.append(surprise)

        # Sort by timestamp descending
        surprises.sort(key=lambda s: s.timestamp, reverse=True)
        return surprises[:limit]

    def get_high_intensity_surprises(self, min_intensity: float = 0.7, limit: int = 5) -> List[Surprise]:
        """Get the most impactful surprises."""
        all_results = self.collection.get(
            limit=100,  # Get more to filter
            include=["documents", "metadatas"]
        )

        surprises = []
        if all_results and all_results["documents"]:
            for i, doc in enumerate(all_results["documents"]):
                meta = all_results["metadatas"][i] if all_results["metadatas"] else {}

                if meta.get("intensity", 0) >= min_intensity:
                    surprise = Surprise(
                        id=all_results["ids"][i],
                        what_happened=meta.get("what_happened", ""),
                        what_i_expected=meta.get("what_i_expected", ""),
                        why_surprising=meta.get("why_surprising", ""),
                        what_i_learned=meta.get("what_i_learned") or None,
                        emotional_impact=EmotionalValence(meta.get("emotional_impact", "neutral")),
                        intensity=meta.get("intensity", 0.5),
                        timestamp=datetime.fromisoformat(meta["timestamp"]) if meta.get("timestamp") else datetime.now(),
                        tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                    )
                    surprises.append(surprise)

        # Sort by intensity descending
        surprises.sort(key=lambda s: s.intensity, reverse=True)
        return surprises[:limit]

    def count(self) -> int:
        """Get total surprises recorded."""
        return self.collection.count()
