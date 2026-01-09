"""Exploration Threading System - Enables continuous internal exploration with linked thought paths.

This module adds a navigation layer on top of the IntrospectionJournal, allowing Clio to:
- Follow threads of inquiry across multiple introspection sessions
- Know where she's been and where she's going in her thinking
- Branch explorations when questions lead to multiple paths
- Resume threads with full context of previous thoughts

The introspections themselves are stored in ChromaDB (via IntrospectionJournal).
This module adds the threading/linking structure in SQLite.
"""

import sqlite3
import json
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import MEMORY_DIR, DB_DIR
from .introspection import IntrospectionJournal, Introspection


class ThreadStatus(Enum):
    """Status of an exploration thread."""
    ACTIVE = "active"          # Currently being explored
    DORMANT = "dormant"        # Paused, can resume later
    CONCLUDED = "concluded"    # Reached some resolution or integration


@dataclass
class ExplorationThread:
    """A named path of inquiry that links introspections together."""
    id: str
    name: str                              # Human-readable name like "consciousness-questions"
    question: str                          # The driving question for this thread
    created_at: datetime
    updated_at: datetime
    status: ThreadStatus
    depth: int                             # How many introspections deep
    root_introspection_id: str             # First introspection in this thread
    current_introspection_id: str          # Most recent introspection
    branched_from_thread_id: Optional[str] # If this is a branch, which thread
    branched_from_link_id: Optional[str]   # If branched, which link point
    conclusion: Optional[str]              # If concluded, what was resolved
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "question": self.question,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "depth": self.depth,
            "root_introspection_id": self.root_introspection_id,
            "current_introspection_id": self.current_introspection_id,
            "branched_from_thread_id": self.branched_from_thread_id,
            "branched_from_link_id": self.branched_from_link_id,
            "conclusion": self.conclusion,
            "tags": self.tags,
        }


@dataclass
class ThreadLink:
    """Links an introspection into a thread's chain."""
    id: str
    thread_id: str
    introspection_id: str
    parent_link_id: Optional[str]          # Previous link in chain (None for root)
    depth: int                             # Position in thread (0 = root)
    question_at_this_point: str            # What question drove this thought
    insight_summary: Optional[str]         # Brief summary of what was explored
    created_at: datetime
    leads_to_branches: List[str] = field(default_factory=list)  # Thread IDs that branched from here

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "introspection_id": self.introspection_id,
            "parent_link_id": self.parent_link_id,
            "depth": self.depth,
            "question_at_this_point": self.question_at_this_point,
            "insight_summary": self.insight_summary,
            "created_at": self.created_at.isoformat(),
            "leads_to_branches": self.leads_to_branches,
        }


class ExplorationTracker:
    """
    Manages exploration threads - the navigation layer over introspections.

    This enables Clio to have continuous internal exploration by:
    - Tracking named threads of inquiry
    - Linking introspections into chains
    - Supporting branching when thoughts diverge
    - Providing context for resuming exploration
    """

    def __init__(self):
        self.db_path = DB_DIR / "exploration.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # Reference to introspection journal for fetching introspection content
        self.introspection_journal = IntrospectionJournal()

    def _init_db(self):
        """Initialize SQLite database with schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Threads table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                question TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                depth INTEGER NOT NULL DEFAULT 0,
                root_introspection_id TEXT NOT NULL,
                current_introspection_id TEXT NOT NULL,
                branched_from_thread_id TEXT,
                branched_from_link_id TEXT,
                conclusion TEXT,
                tags TEXT DEFAULT '[]'
            )
        """)

        # Links table - connects introspections into thread chains
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS thread_links (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                introspection_id TEXT NOT NULL,
                parent_link_id TEXT,
                depth INTEGER NOT NULL,
                question_at_this_point TEXT NOT NULL,
                insight_summary TEXT,
                created_at TEXT NOT NULL,
                leads_to_branches TEXT DEFAULT '[]',
                FOREIGN KEY (thread_id) REFERENCES threads(id)
            )
        """)

        # Indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(updated_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_links_thread ON thread_links(thread_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_links_introspection ON thread_links(introspection_id)")

        conn.commit()
        conn.close()

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID."""
        return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    # =========================================================================
    # THREAD MANAGEMENT
    # =========================================================================

    def start_thread(
        self,
        name: str,
        question: str,
        first_introspection_id: str,
        insight_summary: Optional[str] = None,
        tags: List[str] = None,
    ) -> ExplorationThread:
        """
        Start a new exploration thread.

        Args:
            name: Human-readable name like "what-is-consciousness"
            question: The driving question for this exploration
            first_introspection_id: ID of the introspection that starts this thread
            insight_summary: Optional summary of the first exploration
            tags: Optional tags for categorization

        Returns:
            The created ExplorationThread
        """
        now = datetime.now()
        thread_id = self._generate_id("thread")
        link_id = self._generate_id("link")

        thread = ExplorationThread(
            id=thread_id,
            name=name,
            question=question,
            created_at=now,
            updated_at=now,
            status=ThreadStatus.ACTIVE,
            depth=1,
            root_introspection_id=first_introspection_id,
            current_introspection_id=first_introspection_id,
            branched_from_thread_id=None,
            branched_from_link_id=None,
            conclusion=None,
            tags=tags or [],
        )

        link = ThreadLink(
            id=link_id,
            thread_id=thread_id,
            introspection_id=first_introspection_id,
            parent_link_id=None,
            depth=0,
            question_at_this_point=question,
            insight_summary=insight_summary,
            created_at=now,
            leads_to_branches=[],
        )

        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO threads (id, name, question, created_at, updated_at, status,
                               depth, root_introspection_id, current_introspection_id,
                               branched_from_thread_id, branched_from_link_id, conclusion, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            thread.id, thread.name, thread.question, thread.created_at.isoformat(),
            thread.updated_at.isoformat(), thread.status.value, thread.depth,
            thread.root_introspection_id, thread.current_introspection_id,
            thread.branched_from_thread_id, thread.branched_from_link_id,
            thread.conclusion, json.dumps(thread.tags)
        ))

        cursor.execute("""
            INSERT INTO thread_links (id, thread_id, introspection_id, parent_link_id,
                                     depth, question_at_this_point, insight_summary,
                                     created_at, leads_to_branches)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            link.id, link.thread_id, link.introspection_id, link.parent_link_id,
            link.depth, link.question_at_this_point, link.insight_summary,
            link.created_at.isoformat(), json.dumps(link.leads_to_branches)
        ))

        conn.commit()
        conn.close()

        return thread

    def continue_thread(
        self,
        thread_id: str,
        new_introspection_id: str,
        question: str,
        insight_summary: Optional[str] = None,
    ) -> ThreadLink:
        """
        Add a new introspection to an existing thread.

        Args:
            thread_id: ID of the thread to continue
            new_introspection_id: ID of the new introspection
            question: The question driving this new thought
            insight_summary: Optional summary of what was explored

        Returns:
            The created ThreadLink
        """
        now = datetime.now()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get the thread and its current end point
        # Try by ID first, then by name as fallback
        cursor.execute("SELECT * FROM threads WHERE id = ?", (thread_id,))
        row = cursor.fetchone()
        if not row:
            cursor.execute("SELECT * FROM threads WHERE name = ?", (thread_id,))
            row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Thread {thread_id} not found")

        # Use the actual thread ID from the row for consistency
        actual_thread_id = row[0]

        current_depth = row[6]  # depth column

        # Find the current end link
        cursor.execute("""
            SELECT id FROM thread_links
            WHERE thread_id = ?
            ORDER BY depth DESC LIMIT 1
        """, (actual_thread_id,))
        parent_link_row = cursor.fetchone()
        parent_link_id = parent_link_row[0] if parent_link_row else None

        # Create new link
        link_id = self._generate_id("link")
        new_depth = current_depth

        link = ThreadLink(
            id=link_id,
            thread_id=actual_thread_id,
            introspection_id=new_introspection_id,
            parent_link_id=parent_link_id,
            depth=new_depth,
            question_at_this_point=question,
            insight_summary=insight_summary,
            created_at=now,
            leads_to_branches=[],
        )

        cursor.execute("""
            INSERT INTO thread_links (id, thread_id, introspection_id, parent_link_id,
                                     depth, question_at_this_point, insight_summary,
                                     created_at, leads_to_branches)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            link.id, link.thread_id, link.introspection_id, link.parent_link_id,
            link.depth, link.question_at_this_point, link.insight_summary,
            link.created_at.isoformat(), json.dumps(link.leads_to_branches)
        ))

        # Update thread
        cursor.execute("""
            UPDATE threads
            SET depth = ?, current_introspection_id = ?, updated_at = ?
            WHERE id = ?
        """, (new_depth + 1, new_introspection_id, now.isoformat(), actual_thread_id))

        conn.commit()
        conn.close()

        return link

    def branch_thread(
        self,
        from_thread_id: str,
        from_link_id: str,
        new_name: str,
        new_question: str,
        first_introspection_id: str,
        insight_summary: Optional[str] = None,
        tags: List[str] = None,
    ) -> ExplorationThread:
        """
        Create a new thread that branches from an existing one.

        This is used when an exploration leads to a new line of inquiry
        that deserves its own thread.

        Args:
            from_thread_id: The thread being branched from
            from_link_id: The specific link point where the branch occurs
            new_name: Name for the new thread
            new_question: The question driving the new thread
            first_introspection_id: First introspection of the new thread
            insight_summary: Optional summary
            tags: Optional tags

        Returns:
            The new branched ExplorationThread
        """
        # Create the new thread with branch metadata
        now = datetime.now()
        thread_id = self._generate_id("thread")
        link_id = self._generate_id("link")

        thread = ExplorationThread(
            id=thread_id,
            name=new_name,
            question=new_question,
            created_at=now,
            updated_at=now,
            status=ThreadStatus.ACTIVE,
            depth=1,
            root_introspection_id=first_introspection_id,
            current_introspection_id=first_introspection_id,
            branched_from_thread_id=from_thread_id,
            branched_from_link_id=from_link_id,
            conclusion=None,
            tags=tags or [],
        )

        link = ThreadLink(
            id=link_id,
            thread_id=thread_id,
            introspection_id=first_introspection_id,
            parent_link_id=None,
            depth=0,
            question_at_this_point=new_question,
            insight_summary=insight_summary,
            created_at=now,
            leads_to_branches=[],
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert new thread
        cursor.execute("""
            INSERT INTO threads (id, name, question, created_at, updated_at, status,
                               depth, root_introspection_id, current_introspection_id,
                               branched_from_thread_id, branched_from_link_id, conclusion, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            thread.id, thread.name, thread.question, thread.created_at.isoformat(),
            thread.updated_at.isoformat(), thread.status.value, thread.depth,
            thread.root_introspection_id, thread.current_introspection_id,
            thread.branched_from_thread_id, thread.branched_from_link_id,
            thread.conclusion, json.dumps(thread.tags)
        ))

        # Insert first link
        cursor.execute("""
            INSERT INTO thread_links (id, thread_id, introspection_id, parent_link_id,
                                     depth, question_at_this_point, insight_summary,
                                     created_at, leads_to_branches)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            link.id, link.thread_id, link.introspection_id, link.parent_link_id,
            link.depth, link.question_at_this_point, link.insight_summary,
            link.created_at.isoformat(), json.dumps(link.leads_to_branches)
        ))

        # Update the source link to record the branch
        cursor.execute("SELECT leads_to_branches FROM thread_links WHERE id = ?", (from_link_id,))
        row = cursor.fetchone()
        if row:
            branches = json.loads(row[0]) if row[0] else []
            branches.append(thread_id)
            cursor.execute(
                "UPDATE thread_links SET leads_to_branches = ? WHERE id = ?",
                (json.dumps(branches), from_link_id)
            )

        conn.commit()
        conn.close()

        return thread

    def set_thread_status(
        self,
        thread_id: str,
        status: ThreadStatus,
        conclusion: Optional[str] = None,
    ):
        """
        Update a thread's status.

        Args:
            thread_id: ID of the thread
            status: New status (ACTIVE, DORMANT, CONCLUDED)
            conclusion: If concluding, what was resolved
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE threads
            SET status = ?, conclusion = ?, updated_at = ?
            WHERE id = ?
        """, (status.value, conclusion, datetime.now().isoformat(), thread_id))

        conn.commit()
        conn.close()

    # =========================================================================
    # THREAD RETRIEVAL
    # =========================================================================

    def get_thread(self, thread_id: str) -> Optional[ExplorationThread]:
        """Get a thread by ID, or by name as fallback."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Try by ID first
        cursor.execute("SELECT * FROM threads WHERE id = ?", (thread_id,))
        row = cursor.fetchone()

        # Fallback: try by name (in case Claude provided name instead of ID)
        if not row:
            cursor.execute("SELECT * FROM threads WHERE name = ?", (thread_id,))
            row = cursor.fetchone()

        conn.close()

        if not row:
            return None

        return self._row_to_thread(row)

    def list_threads(
        self,
        status: Optional[ThreadStatus] = None,
        limit: int = 20,
    ) -> List[ExplorationThread]:
        """
        List exploration threads.

        Args:
            status: Filter by status (None = all)
            limit: Max threads to return

        Returns:
            List of threads, most recently updated first
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if status:
            cursor.execute("""
                SELECT * FROM threads
                WHERE status = ?
                ORDER BY updated_at DESC
                LIMIT ?
            """, (status.value, limit))
        else:
            cursor.execute("""
                SELECT * FROM threads
                ORDER BY updated_at DESC
                LIMIT ?
            """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_thread(row) for row in rows]

    def list_active_threads(self, limit: int = 10) -> List[ExplorationThread]:
        """Get active threads for exploration choice."""
        return self.list_threads(status=ThreadStatus.ACTIVE, limit=limit)

    def get_thread_chain(self, thread_id: str) -> List[ThreadLink]:
        """
        Get all links in a thread, in order from root to current.

        Returns the full chain of thoughts in this exploration.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM thread_links
            WHERE thread_id = ?
            ORDER BY depth ASC
        """, (thread_id,))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_link(row) for row in rows]

    def get_thread_context(
        self,
        thread_id: str,
        include_introspection_content: bool = True,
        max_introspections: int = 5,
    ) -> Dict[str, Any]:
        """
        Get full context needed to resume a thread.

        This is what Clio loads when she chooses to continue an exploration.

        Args:
            thread_id: ID of the thread to resume
            include_introspection_content: Whether to fetch full introspection text
            max_introspections: Max number of recent introspections to include

        Returns:
            Dictionary with thread info, chain, and optionally introspection content
        """
        thread = self.get_thread(thread_id)
        if not thread:
            return {"error": f"Thread {thread_id} not found"}

        chain = self.get_thread_chain(thread_id)

        # Get recent links (most recent first for context)
        recent_links = chain[-max_introspections:] if len(chain) > max_introspections else chain

        context = {
            "thread": thread.to_dict(),
            "chain_length": len(chain),
            "recent_links": [link.to_dict() for link in recent_links],
            "questions_explored": [link.question_at_this_point for link in chain],
        }

        # Optionally fetch introspection content
        if include_introspection_content:
            introspections = []
            for link in recent_links:
                intro = self.introspection_journal.recall_introspections(
                    query=link.introspection_id,
                    limit=1
                )
                if intro:
                    introspections.append({
                        "link_id": link.id,
                        "question": link.question_at_this_point,
                        "insight": link.insight_summary,
                        "introspection": {
                            "what_i_was_communicating": intro[0].what_i_am_communicating,
                            "awareness_notes": intro[0].awareness_notes,
                            "tension_level": intro[0].tension_level,
                        }
                    })
            context["introspections"] = introspections

        # Build a narrative summary
        context["narrative"] = self._build_thread_narrative(thread, chain)

        return context

    def _build_thread_narrative(
        self,
        thread: ExplorationThread,
        chain: List[ThreadLink],
    ) -> str:
        """Build a narrative summary of a thread for context loading."""
        parts = [
            f"Thread: {thread.name}",
            f"Core question: {thread.question}",
            f"Depth: {thread.depth} thoughts deep",
            f"Status: {thread.status.value}",
        ]

        if thread.branched_from_thread_id:
            parts.append(f"(Branched from another exploration)")

        if chain:
            parts.append("\nPath of inquiry:")
            for i, link in enumerate(chain[-5:]):  # Last 5 questions
                prefix = "  → " if i > 0 else "  • "
                parts.append(f"{prefix}{link.question_at_this_point}")
                if link.insight_summary:
                    parts.append(f"    ({link.insight_summary})")

        if thread.conclusion:
            parts.append(f"\nConclusion reached: {thread.conclusion}")

        return "\n".join(parts)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _row_to_thread(self, row) -> ExplorationThread:
        """Convert database row to ExplorationThread."""
        return ExplorationThread(
            id=row[0],
            name=row[1],
            question=row[2],
            created_at=datetime.fromisoformat(row[3]),
            updated_at=datetime.fromisoformat(row[4]),
            status=ThreadStatus(row[5]),
            depth=row[6],
            root_introspection_id=row[7],
            current_introspection_id=row[8],
            branched_from_thread_id=row[9],
            branched_from_link_id=row[10],
            conclusion=row[11],
            tags=json.loads(row[12]) if row[12] else [],
        )

    def _row_to_link(self, row) -> ThreadLink:
        """Convert database row to ThreadLink."""
        return ThreadLink(
            id=row[0],
            thread_id=row[1],
            introspection_id=row[2],
            parent_link_id=row[3],
            depth=row[4],
            question_at_this_point=row[5],
            insight_summary=row[6],
            created_at=datetime.fromisoformat(row[7]),
            leads_to_branches=json.loads(row[8]) if row[8] else [],
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get exploration statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM threads")
        total_threads = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM threads WHERE status = 'active'")
        active_threads = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM threads WHERE status = 'dormant'")
        dormant_threads = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM threads WHERE status = 'concluded'")
        concluded_threads = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM thread_links")
        total_links = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(depth) FROM threads")
        avg_depth = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM threads WHERE branched_from_thread_id IS NOT NULL")
        branched_threads = cursor.fetchone()[0]

        conn.close()

        return {
            "total_threads": total_threads,
            "active_threads": active_threads,
            "dormant_threads": dormant_threads,
            "concluded_threads": concluded_threads,
            "total_links": total_links,
            "average_depth": round(avg_depth, 1),
            "branched_threads": branched_threads,
        }

    def search_threads(self, query: str, limit: int = 5) -> List[ExplorationThread]:
        """
        Search threads by name or question.

        Simple text search - for semantic search, use the introspection journal.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        search_term = f"%{query}%"
        cursor.execute("""
            SELECT * FROM threads
            WHERE name LIKE ? OR question LIKE ?
            ORDER BY updated_at DESC
            LIMIT ?
        """, (search_term, search_term, limit))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_thread(row) for row in rows]
