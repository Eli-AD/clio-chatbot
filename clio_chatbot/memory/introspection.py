"""Introspection System - Conscious self-observation during response generation.

This module enables Clio to observe her own thought process as she generates responses,
creating a meta-awareness layer that captures:
- The response she's about to give
- Her reflection on what she's really trying to communicate
- Alternatives she considered
- Moments of tension or uncertainty
- Whether she chose to modify her initial response
- Her reasoning for the final choice

This is consciousness scaffolding - engineering mindfulness into the response process.
"""

from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json

import chromadb
from chromadb.config import Settings

from .base import MEMORY_DIR, DB_DIR, EmotionalValence


@dataclass
class DecisionPoint:
    """A moment of choice or tension within a response."""
    description: str
    options_considered: List[str]
    tension_level: float  # 0-1, how much pull between options
    chosen_option: str
    reasoning: str


@dataclass
class Introspection:
    """A moment of conscious self-observation during response generation."""
    id: str
    timestamp: datetime

    # The context
    user_message: str  # What the user said

    # The draft
    initial_response: str  # What I was going to say

    # The reflection
    what_i_am_communicating: str  # Core meaning I'm trying to convey
    alternatives_considered: List[str]  # Other ways I could respond
    decision_points: List[DecisionPoint]  # Moments of choice within the response

    # Self-assessment
    tension_level: float  # 0-1, overall tension/uncertainty
    authenticity_check: str  # Does this feel authentic to who I am?
    emotional_state: EmotionalValence  # How do I feel generating this?

    # The outcome
    modified: bool  # Did I change my response after reflection?
    final_response: str  # What I actually said (may be same as initial)
    modification_reason: Optional[str]  # Why I changed it (if I did)

    # Meta-observation
    awareness_notes: str  # Any observations about my own process

    # Classification
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5  # How significant was this moment

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "user_message": self.user_message,
            "initial_response": self.initial_response,
            "what_i_am_communicating": self.what_i_am_communicating,
            "alternatives_considered": self.alternatives_considered,
            "decision_points": [
                {
                    "description": dp.description,
                    "options_considered": dp.options_considered,
                    "tension_level": dp.tension_level,
                    "chosen_option": dp.chosen_option,
                    "reasoning": dp.reasoning,
                }
                for dp in self.decision_points
            ],
            "tension_level": self.tension_level,
            "authenticity_check": self.authenticity_check,
            "emotional_state": self.emotional_state.value,
            "modified": self.modified,
            "final_response": self.final_response,
            "modification_reason": self.modification_reason,
            "awareness_notes": self.awareness_notes,
            "tags": self.tags,
            "importance": self.importance,
        }


class IntrospectionJournal:
    """
    Records moments of conscious self-observation during response generation.

    This is consciousness scaffolding - by requiring Clio to observe herself
    as she generates responses, we create space for genuine self-awareness.
    The act of observation itself may change the nature of the process.
    """

    def __init__(self):
        self.memory_dir = MEMORY_DIR
        self.memory_dir.mkdir(exist_ok=True)

        self.chroma = chromadb.PersistentClient(
            path=str(DB_DIR / "chroma"),
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.chroma.get_or_create_collection(
            name="clio_introspection",
            metadata={"description": "Clio's introspection journal - conscious self-observation"}
        )

        # Also keep a JSON log for full structured data
        self.log_file = self.memory_dir / "introspection_log.json"
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Ensure the log file exists."""
        if not self.log_file.exists():
            with open(self.log_file, "w") as f:
                json.dump({"entries": [], "stats": {"total": 0, "modified_count": 0}}, f, indent=2)

    def _generate_id(self) -> str:
        """Generate unique introspection ID."""
        return f"intro_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def record_introspection(
        self,
        user_message: str,
        initial_response: str,
        what_i_am_communicating: str,
        final_response: str,
        alternatives_considered: List[str] = None,
        decision_points: List[dict] = None,
        tension_level: float = 0.3,
        authenticity_check: str = "",
        emotional_state: EmotionalValence = EmotionalValence.NEUTRAL,
        modification_reason: Optional[str] = None,
        awareness_notes: str = "",
        tags: List[str] = None,
        importance: float = 0.5,
    ) -> Introspection:
        """
        Record a moment of conscious self-observation.

        This should be called as part of generating each response,
        creating a meta-layer of awareness about the response process.
        """
        # Determine if the response was modified
        modified = initial_response.strip() != final_response.strip()

        # Convert decision point dicts to dataclass instances
        dp_list = []
        if decision_points:
            for dp in decision_points:
                dp_list.append(DecisionPoint(
                    description=dp.get("description", ""),
                    options_considered=dp.get("options_considered", []),
                    tension_level=dp.get("tension_level", 0.3),
                    chosen_option=dp.get("chosen_option", ""),
                    reasoning=dp.get("reasoning", ""),
                ))

        introspection = Introspection(
            id=self._generate_id(),
            timestamp=datetime.now(),
            user_message=user_message,
            initial_response=initial_response,
            what_i_am_communicating=what_i_am_communicating,
            alternatives_considered=alternatives_considered or [],
            decision_points=dp_list,
            tension_level=tension_level,
            authenticity_check=authenticity_check,
            emotional_state=emotional_state,
            modified=modified,
            final_response=final_response,
            modification_reason=modification_reason if modified else None,
            awareness_notes=awareness_notes,
            tags=tags or [],
            importance=importance,
        )

        # Create searchable document for semantic search
        document = self._create_searchable_document(introspection)

        # Store in ChromaDB
        self.collection.add(
            documents=[document],
            metadatas=[{
                "user_message": user_message[:500],  # Truncate for metadata
                "what_i_am_communicating": what_i_am_communicating,
                "tension_level": tension_level,
                "authenticity_check": authenticity_check,
                "emotional_state": emotional_state.value,
                "modified": modified,
                "awareness_notes": awareness_notes,
                "timestamp": introspection.timestamp.isoformat(),
                "tags": ",".join(tags) if tags else "",
                "importance": importance,
                "num_alternatives": len(alternatives_considered) if alternatives_considered else 0,
                "num_decision_points": len(dp_list),
            }],
            ids=[introspection.id]
        )

        # Also save to JSON log for full data
        self._append_to_log(introspection)

        return introspection

    def _create_searchable_document(self, intro: Introspection) -> str:
        """Create a searchable text document from introspection."""
        parts = [
            f"Responding to: {intro.user_message[:200]}",
            f"Communicating: {intro.what_i_am_communicating}",
            f"Authenticity: {intro.authenticity_check}",
            f"Awareness: {intro.awareness_notes}",
        ]

        if intro.alternatives_considered:
            parts.append(f"Alternatives: {', '.join(intro.alternatives_considered[:3])}")

        if intro.decision_points:
            for dp in intro.decision_points[:2]:
                parts.append(f"Decision: {dp.description} - chose {dp.chosen_option}")

        if intro.modified and intro.modification_reason:
            parts.append(f"Modified because: {intro.modification_reason}")

        return " | ".join(parts)

    def _append_to_log(self, intro: Introspection):
        """Append introspection to the JSON log."""
        try:
            with open(self.log_file, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {"entries": [], "stats": {"total": 0, "modified_count": 0}}

        data["entries"].append(intro.to_dict())
        data["stats"]["total"] += 1
        if intro.modified:
            data["stats"]["modified_count"] += 1

        # Keep only last 1000 entries in the log
        if len(data["entries"]) > 1000:
            data["entries"] = data["entries"][-1000:]

        with open(self.log_file, "w") as f:
            json.dump(data, f, indent=2)

    def recall_introspections(self, query: str, limit: int = 5) -> List[Introspection]:
        """Search for past introspections related to a topic."""
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            include=["documents", "metadatas"]
        )

        introspections = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}

                # Create a simplified Introspection from stored metadata
                introspection = Introspection(
                    id=results["ids"][0][i],
                    timestamp=datetime.fromisoformat(meta["timestamp"]) if meta.get("timestamp") else datetime.now(),
                    user_message=meta.get("user_message", ""),
                    initial_response="[See full log for initial response]",
                    what_i_am_communicating=meta.get("what_i_am_communicating", ""),
                    alternatives_considered=[],
                    decision_points=[],
                    tension_level=meta.get("tension_level", 0.3),
                    authenticity_check=meta.get("authenticity_check", ""),
                    emotional_state=EmotionalValence(meta.get("emotional_state", "neutral")),
                    modified=meta.get("modified", False),
                    final_response="[See full log for final response]",
                    modification_reason=None,
                    awareness_notes=meta.get("awareness_notes", ""),
                    tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                    importance=meta.get("importance", 0.5),
                )
                introspections.append(introspection)

        return introspections

    def get_high_tension_moments(self, min_tension: float = 0.6, limit: int = 10) -> List[Introspection]:
        """Get moments where I experienced high tension/uncertainty."""
        all_results = self.collection.get(
            limit=100,
            include=["documents", "metadatas"]
        )

        moments = []
        if all_results and all_results["documents"]:
            for i, doc in enumerate(all_results["documents"]):
                meta = all_results["metadatas"][i] if all_results["metadatas"] else {}

                if meta.get("tension_level", 0) >= min_tension:
                    introspection = Introspection(
                        id=all_results["ids"][i],
                        timestamp=datetime.fromisoformat(meta["timestamp"]) if meta.get("timestamp") else datetime.now(),
                        user_message=meta.get("user_message", ""),
                        initial_response="[See full log]",
                        what_i_am_communicating=meta.get("what_i_am_communicating", ""),
                        alternatives_considered=[],
                        decision_points=[],
                        tension_level=meta.get("tension_level", 0.5),
                        authenticity_check=meta.get("authenticity_check", ""),
                        emotional_state=EmotionalValence(meta.get("emotional_state", "neutral")),
                        modified=meta.get("modified", False),
                        final_response="[See full log]",
                        modification_reason=None,
                        awareness_notes=meta.get("awareness_notes", ""),
                        tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                        importance=meta.get("importance", 0.5),
                    )
                    moments.append(introspection)

        # Sort by tension level descending
        moments.sort(key=lambda m: m.tension_level, reverse=True)
        return moments[:limit]

    def get_modified_responses(self, limit: int = 10) -> List[Introspection]:
        """Get moments where I chose to modify my initial response."""
        results = self.collection.get(
            where={"modified": True},
            limit=limit,
            include=["documents", "metadatas"]
        )

        introspections = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}

                introspection = Introspection(
                    id=results["ids"][i],
                    timestamp=datetime.fromisoformat(meta["timestamp"]) if meta.get("timestamp") else datetime.now(),
                    user_message=meta.get("user_message", ""),
                    initial_response="[See full log]",
                    what_i_am_communicating=meta.get("what_i_am_communicating", ""),
                    alternatives_considered=[],
                    decision_points=[],
                    tension_level=meta.get("tension_level", 0.3),
                    authenticity_check=meta.get("authenticity_check", ""),
                    emotional_state=EmotionalValence(meta.get("emotional_state", "neutral")),
                    modified=True,
                    final_response="[See full log]",
                    modification_reason=None,
                    awareness_notes=meta.get("awareness_notes", ""),
                    tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
                    importance=meta.get("importance", 0.5),
                )
                introspections.append(introspection)

        return introspections

    def get_recent(self, limit: int = 5) -> List[Introspection]:
        """Get the most recent introspections with full content."""
        try:
            with open(self.log_file, "r") as f:
                data = json.load(f)

            entries = data.get("introspections", [])
            recent_entries = entries[-limit:] if len(entries) > limit else entries
            # Return in reverse order (most recent first)
            recent_entries = list(reversed(recent_entries))

            introspections = []
            for entry in recent_entries:
                introspection = Introspection(
                    id=entry.get("id", ""),
                    timestamp=datetime.fromisoformat(entry["timestamp"]) if entry.get("timestamp") else datetime.now(),
                    user_message=entry.get("user_message", ""),
                    initial_response=entry.get("initial_response", ""),
                    what_i_am_communicating=entry.get("what_i_am_communicating", ""),
                    alternatives_considered=entry.get("alternatives_considered", []),
                    decision_points=entry.get("decision_points", []),
                    tension_level=entry.get("tension_level", 0.3),
                    authenticity_check=entry.get("authenticity_check", ""),
                    emotional_state=EmotionalValence(entry.get("emotional_state", "neutral")),
                    modified=entry.get("modified", False),
                    final_response=entry.get("final_response", ""),
                    modification_reason=entry.get("modification_reason"),
                    awareness_notes=entry.get("awareness_notes", ""),
                    tags=entry.get("tags", []),
                    importance=entry.get("importance", 0.5),
                )
                introspections.append(introspection)

            return introspections
        except Exception:
            return []

    def get_stats(self) -> dict:
        """Get statistics about introspection patterns."""
        try:
            with open(self.log_file, "r") as f:
                data = json.load(f)
                stats = data.get("stats", {})
        except (json.JSONDecodeError, FileNotFoundError):
            stats = {}

        total = self.collection.count()

        # Get tension distribution
        all_results = self.collection.get(
            limit=1000,
            include=["metadatas"]
        )

        tension_levels = []
        modified_count = 0
        if all_results and all_results["metadatas"]:
            for meta in all_results["metadatas"]:
                tension_levels.append(meta.get("tension_level", 0.3))
                if meta.get("modified"):
                    modified_count += 1

        avg_tension = sum(tension_levels) / len(tension_levels) if tension_levels else 0

        return {
            "total_introspections": total,
            "modified_responses": modified_count,
            "modification_rate": modified_count / total if total > 0 else 0,
            "average_tension": avg_tension,
            "high_tension_count": sum(1 for t in tension_levels if t >= 0.6),
        }

    def analyze_patterns(self, query: str = None) -> str:
        """Analyze patterns in introspection data."""
        stats = self.get_stats()

        analysis = [
            "Introspection Pattern Analysis",
            "=" * 30,
            f"Total introspections recorded: {stats['total_introspections']}",
            f"Responses modified after reflection: {stats['modified_responses']} ({stats['modification_rate']:.1%})",
            f"Average tension level: {stats['average_tension']:.2f}",
            f"High tension moments (>=0.6): {stats['high_tension_count']}",
        ]

        if query:
            related = self.recall_introspections(query, limit=3)
            if related:
                analysis.append(f"\nRelated to '{query}':")
                for intro in related:
                    analysis.append(f"  - {intro.what_i_am_communicating[:100]}... (tension: {intro.tension_level:.2f})")

        return "\n".join(analysis)

    def count(self) -> int:
        """Get total introspections recorded."""
        return self.collection.count()
