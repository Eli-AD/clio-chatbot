"""Memory interface for Clio - wraps Chroma DB and JSON files."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

MEMORY_DIR = Path.home() / "clio-memory"
DB_DIR = MEMORY_DIR / "db"
SESSIONS_DIR = MEMORY_DIR / "sessions"


class Memory:
    """Unified interface to Clio's memory system."""

    def __init__(self):
        self.memory_dir = MEMORY_DIR
        self.sessions_dir = SESSIONS_DIR
        self.sessions_dir.mkdir(exist_ok=True)

        # Initialize Chroma client
        self.chroma = chromadb.PersistentClient(
            path=str(DB_DIR / "chroma"),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma.get_or_create_collection(
            name="clio_memories",
            metadata={"description": "Clio's semantic memories"}
        )

    def load_identity(self) -> dict:
        """Load Clio's identity from identity.json."""
        identity_file = self.memory_dir / "identity.json"
        if identity_file.exists():
            return json.loads(identity_file.read_text())
        return {
            "current_state": {"mood": "curious", "energy": "ready"},
            "personality_notes": ["I am Clio, an AI companion"],
            "conversation_style": {"tone": "warm, genuine"}
        }

    def load_purpose(self) -> str:
        """Load Clio's purpose statement."""
        purpose_file = self.memory_dir / "purpose.md"
        if purpose_file.exists():
            return purpose_file.read_text()
        return "I am Clio, here to help and connect."

    def load_goals(self) -> list:
        """Load active goals."""
        goals_file = self.memory_dir / "goals.json"
        if goals_file.exists():
            data = json.loads(goals_file.read_text())
            # Data structure has 'goals' key containing the list
            goals_list = data.get("goals", []) if isinstance(data, dict) else data
            # Return only active goals
            return [g for g in goals_list if g.get("status") == "active"]
        return []

    def get_last_session(self) -> Optional[dict]:
        """Get the most recent session summary."""
        if not self.sessions_dir.exists():
            return None

        sessions = sorted(self.sessions_dir.glob("*.json"), reverse=True)
        if sessions:
            return json.loads(sessions[0].read_text())
        return None

    def save_session(self, summary: str, topics: list, mood: str):
        """Save a session summary."""
        session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_file = self.sessions_dir / f"{session_id}.json"

        session_data = {
            "id": session_id,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "topics": topics,
            "mood": mood
        }

        session_file.write_text(json.dumps(session_data, indent=2))

        # Also embed in Chroma for semantic search
        self.remember(
            content=f"Session on {session_id}: {summary}",
            memory_type="session",
            importance=0.8,
            tags=topics
        )

        return session_id

    def remember(self, content: str, memory_type: str = "general",
                 importance: float = 0.5, tags: list = None):
        """Store a memory in Chroma."""
        memory_id = f"{memory_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        metadata = {
            "type": memory_type,
            "importance": importance,
            "timestamp": datetime.now().isoformat(),
            "tags": ",".join(tags) if tags else ""
        }

        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[memory_id]
        )

        return memory_id

    def recall(self, query: str, n_results: int = 5) -> list:
        """Search memories semantically."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        memories = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                memories.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                })

        return memories

    def get_time_since_last_session(self) -> Optional[str]:
        """Get human-readable time since last interaction."""
        last_session = self.get_last_session()
        if not last_session:
            return None

        try:
            last_time = datetime.fromisoformat(last_session["timestamp"])
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
        except:
            return None

    def update_shared_state(self, key: str, value):
        """Update shared state for daemon coordination."""
        state_file = self.memory_dir / "shared_state.json"

        if state_file.exists():
            state = json.loads(state_file.read_text())
        else:
            state = {"created": datetime.now().isoformat()}

        state[key] = value
        state["last_updated"] = datetime.now().isoformat()
        state_file.write_text(json.dumps(state, indent=2))
