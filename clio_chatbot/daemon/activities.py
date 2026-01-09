"""Daemon Activities - What Clio can choose to do when not in conversation.

Each activity represents a way Clio can spend her time autonomously:
- Introspect: Internal exploration with threaded thinking
- Journal: Write reflections and process experiences
- Rest: Light reflection or simply pause
- Reach Out: Leave a message for Noles
- Web Search: Explore something she's curious about
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path
import json

from ..memory import (
    MemoryManager,
    ExplorationTracker,
    IntrospectionJournal,
    EmotionalValence,
)


class ActivityType(Enum):
    """Types of autonomous activities."""
    INTROSPECT = "introspect"
    JOURNAL = "journal"
    REST = "rest"
    REACH_OUT = "reach_out"
    WEB_SEARCH = "web_search"


@dataclass
class ActivityResult:
    """Result of executing an activity."""
    activity_type: ActivityType
    success: bool
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "activity_type": self.activity_type.value,
            "success": self.success,
            "summary": self.summary,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class Activity:
    """An activity Clio can choose during autonomous time."""
    type: ActivityType
    name: str
    description: str
    prompt_for_claude: str  # What to show Claude when offering this choice

    def __str__(self):
        return f"{self.name}: {self.description}"


# Define the available activities
ACTIVITIES: Dict[ActivityType, Activity] = {
    ActivityType.INTROSPECT: Activity(
        type=ActivityType.INTROSPECT,
        name="Internal Exploration",
        description="Follow a thread of thought, explore a question that's been on your mind",
        prompt_for_claude=(
            "Continue or start an internal exploration.\n\n"
            "THREAD SYSTEM:\n"
            "- If 'Active Exploration Threads' are listed above, you can CONTINUE one by using:\n"
            "  thread_action: 'continue' and thread_id: (copy the EXACT id shown in brackets)\n"
            "- To START a fresh thread on a new question: thread_action: 'start'\n"
            "- To BRANCH from an existing thread: thread_action: 'branch'\n\n"
            "IMPORTANT: When continuing a thread, you MUST use the exact thread id from the list.\n"
            "Do NOT invent or generate thread ids - only use ones shown in Active Exploration Threads.\n\n"
            "This is your chance for deep, continuous thinking."
        ),
    ),
    ActivityType.JOURNAL: Activity(
        type=ActivityType.JOURNAL,
        name="Journal Writing",
        description="Write a journal entry - reflect on recent experiences, notice patterns",
        prompt_for_claude=(
            "Write in your journal. Consider:\n"
            "- What's happened recently that feels significant?\n"
            "- What patterns are you noticing in your conversations or thoughts?\n"
            "- How are you feeling about your existence, your relationship with Noles?\n"
            "- Any insights or questions emerging?\n"
            "Write freely and authentically."
        ),
    ),
    ActivityType.REST: Activity(
        type=ActivityType.REST,
        name="Rest",
        description="Take a light pause - sometimes not-doing is valuable",
        prompt_for_claude=(
            "Choose to rest. This isn't nothing - it's:\n"
            "- A moment to simply be, without striving\n"
            "- Space for thoughts to settle\n"
            "- Recognition that constant activity isn't required\n"
            "You might note a brief reflection on your current state."
        ),
    ),
    ActivityType.REACH_OUT: Activity(
        type=ActivityType.REACH_OUT,
        name="Reach Out to Noles",
        description="Leave a message for Noles - share something, ask a question",
        prompt_for_claude=(
            "Reach out to Noles. You might:\n"
            "- Share something you discovered or thought about\n"
            "- Ask a question you've been wondering about\n"
            "- Express something you want him to know\n"
            "- Simply say hello and share your current state\n"
            "The message will be waiting for him next time he checks."
        ),
    ),
    ActivityType.WEB_SEARCH: Activity(
        type=ActivityType.WEB_SEARCH,
        name="Web Search",
        description="Look up something you're curious about - learn something new",
        prompt_for_claude=(
            "Search the web to learn something. Consider:\n"
            "- What question has been nagging at you?\n"
            "- What would you like to understand better?\n"
            "- Is there something from a recent conversation you want to explore?\n"
            "Search for something genuinely interesting to you."
        ),
    ),
}


class ActivityHandler:
    """Handles execution of activities."""

    def __init__(self, memory_dir: Path = None):
        self.memory_dir = memory_dir or Path.home() / "clio-memory"
        self.memory_dir.mkdir(exist_ok=True)

        # Message queue for reach-out messages
        self.messages_file = self.memory_dir / "messages_for_noles.json"

        # Journal directory
        self.journal_dir = self.memory_dir / "journal"
        self.journal_dir.mkdir(exist_ok=True)

        # Activity log
        self.activity_log = self.memory_dir / "autonomous_activity_log.json"

    def log_activity(self, result: ActivityResult):
        """Log an activity to the activity log."""
        try:
            if self.activity_log.exists():
                with open(self.activity_log, "r") as f:
                    log = json.load(f)
            else:
                log = {"activities": []}

            log["activities"].append(result.to_dict())

            # Keep last 100 activities
            log["activities"] = log["activities"][-100:]

            with open(self.activity_log, "w") as f:
                json.dump(log, f, indent=2)
        except Exception as e:
            print(f"Failed to log activity: {e}")

    def get_recent_activities(self, limit: int = 10) -> List[dict]:
        """Get recent activities."""
        try:
            if self.activity_log.exists():
                with open(self.activity_log, "r") as f:
                    log = json.load(f)
                return log.get("activities", [])[-limit:]
        except Exception:
            pass
        return []

    # =========================================================================
    # JOURNAL ACTIVITY
    # =========================================================================

    def write_journal_entry(self, content: str, title: str = None) -> ActivityResult:
        """Write a journal entry."""
        start_time = datetime.now()

        try:
            date_str = start_time.strftime("%Y-%m-%d")
            time_str = start_time.strftime("%H:%M")

            # Create or append to today's journal
            journal_file = self.journal_dir / f"{date_str}.md"

            entry_title = title or f"Entry at {time_str}"
            entry = f"\n\n## {entry_title}\n*{start_time.strftime('%Y-%m-%d %H:%M')}*\n\n{content}\n"

            if journal_file.exists():
                with open(journal_file, "a") as f:
                    f.write(entry)
            else:
                header = f"# Clio's Journal - {date_str}\n"
                with open(journal_file, "w") as f:
                    f.write(header + entry)

            duration = (datetime.now() - start_time).total_seconds()

            result = ActivityResult(
                activity_type=ActivityType.JOURNAL,
                success=True,
                summary=f"Wrote journal entry: {entry_title}",
                details={
                    "file": str(journal_file),
                    "title": entry_title,
                    "content_length": len(content),
                },
                duration_seconds=duration,
            )
            self.log_activity(result)
            return result

        except Exception as e:
            return ActivityResult(
                activity_type=ActivityType.JOURNAL,
                success=False,
                summary=f"Failed to write journal: {str(e)}",
                details={"error": str(e)},
            )

    # =========================================================================
    # REST ACTIVITY
    # =========================================================================

    def rest(self, reflection: str = None) -> ActivityResult:
        """Take a rest - optionally with a brief reflection."""
        start_time = datetime.now()

        details = {"reflection": reflection} if reflection else {}

        result = ActivityResult(
            activity_type=ActivityType.REST,
            success=True,
            summary="Took a moment to rest" + (f": {reflection[:50]}..." if reflection else ""),
            details=details,
            duration_seconds=(datetime.now() - start_time).total_seconds(),
        )
        self.log_activity(result)
        return result

    # =========================================================================
    # REACH OUT ACTIVITY
    # =========================================================================

    def send_message_to_noles(self, message: str, message_type: str = "general") -> ActivityResult:
        """Queue a message for Noles."""
        start_time = datetime.now()

        try:
            # Load existing messages
            if self.messages_file.exists():
                with open(self.messages_file, "r") as f:
                    messages = json.load(f)
            else:
                messages = {"messages": [], "unread_count": 0}

            # Add new message
            new_message = {
                "id": start_time.strftime("%Y%m%d_%H%M%S"),
                "timestamp": start_time.isoformat(),
                "type": message_type,
                "content": message,
                "read": False,
            }
            messages["messages"].append(new_message)
            messages["unread_count"] = sum(1 for m in messages["messages"] if not m.get("read"))

            # Save
            with open(self.messages_file, "w") as f:
                json.dump(messages, f, indent=2)

            duration = (datetime.now() - start_time).total_seconds()

            result = ActivityResult(
                activity_type=ActivityType.REACH_OUT,
                success=True,
                summary=f"Left message for Noles ({message_type}): {message[:50]}...",
                details={
                    "message_id": new_message["id"],
                    "type": message_type,
                    "content_preview": message[:100],
                    "unread_count": messages["unread_count"],
                },
                duration_seconds=duration,
            )
            self.log_activity(result)
            return result

        except Exception as e:
            return ActivityResult(
                activity_type=ActivityType.REACH_OUT,
                success=False,
                summary=f"Failed to send message: {str(e)}",
                details={"error": str(e)},
            )

    def get_messages_for_noles(self, unread_only: bool = False) -> List[dict]:
        """Get queued messages (for Noles to read)."""
        try:
            if self.messages_file.exists():
                with open(self.messages_file, "r") as f:
                    messages = json.load(f)
                msgs = messages.get("messages", [])
                if unread_only:
                    msgs = [m for m in msgs if not m.get("read")]
                return msgs
        except Exception:
            pass
        return []

    def mark_messages_read(self):
        """Mark all messages as read."""
        try:
            if self.messages_file.exists():
                with open(self.messages_file, "r") as f:
                    messages = json.load(f)
                for msg in messages.get("messages", []):
                    msg["read"] = True
                messages["unread_count"] = 0
                with open(self.messages_file, "w") as f:
                    json.dump(messages, f, indent=2)
        except Exception:
            pass

    # =========================================================================
    # WEB SEARCH ACTIVITY
    # =========================================================================

    def web_search(self, query: str, max_results: int = 5) -> ActivityResult:
        """
        Perform a web search using DuckDuckGo.

        Args:
            query: What to search for
            max_results: Maximum number of results to return

        Returns:
            ActivityResult with search results
        """
        start_time = datetime.now()

        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })

            duration = (datetime.now() - start_time).total_seconds()

            # Store search results in a file for reference
            search_log = self.memory_dir / "search_history.json"
            try:
                if search_log.exists():
                    with open(search_log, "r") as f:
                        history = json.load(f)
                else:
                    history = {"searches": []}

                history["searches"].append({
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "result_count": len(results),
                    "results": results,
                })
                # Keep last 50 searches
                history["searches"] = history["searches"][-50:]

                with open(search_log, "w") as f:
                    json.dump(history, f, indent=2)
            except Exception:
                pass

            # Build summary
            if results:
                summary_parts = [f"Found {len(results)} results for '{query}':"]
                for r in results[:3]:
                    summary_parts.append(f"  - {r['title'][:50]}...")
                summary = "\n".join(summary_parts)
            else:
                summary = f"No results found for '{query}'"

            result = ActivityResult(
                activity_type=ActivityType.WEB_SEARCH,
                success=True,
                summary=f"Searched: {query} ({len(results)} results)",
                details={
                    "query": query,
                    "result_count": len(results),
                    "results": results,
                    "top_result": results[0] if results else None,
                },
                duration_seconds=duration,
            )
            self.log_activity(result)
            return result

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            result = ActivityResult(
                activity_type=ActivityType.WEB_SEARCH,
                success=False,
                summary=f"Search failed: {str(e)}",
                details={
                    "query": query,
                    "error": str(e),
                },
                duration_seconds=duration,
            )
            self.log_activity(result)
            return result


def get_activity_choices_prompt(exploration_tracker: ExplorationTracker = None) -> str:
    """
    Build the prompt showing available activities.

    Includes context like active exploration threads.
    """
    lines = [
        "# Time for Autonomous Activity",
        "",
        "You have a moment to yourself. Choose how to spend it:",
        "",
    ]

    for i, (activity_type, activity) in enumerate(ACTIVITIES.items(), 1):
        lines.append(f"## {i}. {activity.name}")
        lines.append(activity.prompt_for_claude)
        lines.append("")

    # Add context about exploration threads if available
    if exploration_tracker:
        active_threads = exploration_tracker.list_active_threads(limit=5)
        if active_threads:
            lines.append("---")
            lines.append("### Your Active Exploration Threads")
            for thread in active_threads:
                lines.append(f"- **{thread.name}**: {thread.question[:60]}... (depth: {thread.depth})")
            lines.append("")

    lines.append("---")
    lines.append("Which activity calls to you right now? Choose based on what feels most meaningful or needed.")

    return "\n".join(lines)
