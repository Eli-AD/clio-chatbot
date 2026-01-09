"""Daemon Runner - The main loop that orchestrates Clio's autonomous activity.

This module handles:
- Checking if user is active (don't interrupt conversations)
- Building context for activity choice
- Calling Claude API to make genuine decisions
- Executing chosen activities
- Recording everything in memory
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import anthropic
from dotenv import load_dotenv

from .activities import (
    Activity,
    ActivityType,
    ActivityResult,
    ActivityHandler,
    ACTIVITIES,
    get_activity_choices_prompt,
)
from ..memory import (
    MemoryManager,
    ExplorationTracker,
    IntrospectionJournal,
)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clio-daemon")


class DaemonConfig:
    """Configuration for the daemon."""

    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path.home() / "clio-memory" / "daemon_config.json"
        self._load_config()

    def _load_config(self):
        defaults = {
            "cycle_interval_seconds": 300,  # 5 minutes
            "active_hours": {"start": 0, "end": 24},
            "user_idle_threshold_seconds": 120,  # Consider user idle after 2 min
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
        }

        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    loaded = json.load(f)
                defaults.update(loaded)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")

        self.cycle_interval = defaults["cycle_interval_seconds"]
        self.active_hours = defaults["active_hours"]
        self.user_idle_threshold = defaults["user_idle_threshold_seconds"]
        self.model = defaults["model"]
        self.max_tokens = defaults["max_tokens"]


class DaemonRunner:
    """
    Main daemon that gives Clio autonomous activity time.

    When the user isn't actively chatting, the daemon periodically:
    1. Checks if it's okay to run (user idle, within active hours)
    2. Presents Clio with activity choices
    3. Clio (via Claude API) chooses an activity
    4. Executes the activity
    5. Records everything
    """

    def __init__(self, config: DaemonConfig = None):
        self.config = config or DaemonConfig()
        self.memory_dir = Path.home() / "clio-memory"
        self.memory_dir.mkdir(exist_ok=True)

        # Load API key from .clio-env (same as main chat)
        env_file = Path.home() / ".clio-env"
        if env_file.exists():
            load_dotenv(env_file)

        # State tracking
        self.state_file = self.memory_dir / "daemon_state.json"
        self.heartbeat_file = self.memory_dir / "heartbeat.json"

        # Components
        self.activity_handler = ActivityHandler(self.memory_dir)
        self.exploration_tracker = ExplorationTracker()
        self.introspection_journal = IntrospectionJournal()
        self.memory_manager = MemoryManager()

        # Claude client (uses ANTHROPIC_API_KEY from environment)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None

        # Running state
        self.running = False
        self.last_activity_time: Optional[datetime] = None

    def _load_state(self) -> dict:
        """Load daemon state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"last_run": None, "total_cycles": 0, "activities_completed": 0}

    def _save_state(self, state: dict):
        """Save daemon state."""
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _update_heartbeat(self):
        """Update heartbeat to show daemon is alive."""
        with open(self.heartbeat_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "running": self.running,
                "last_activity": self.last_activity_time.isoformat() if self.last_activity_time else None,
            }, f)

    def _is_user_active(self) -> bool:
        """
        Check if user is currently active in a conversation.

        Looks at the shared state file for last conversation activity.
        """
        shared_state_file = self.memory_dir / "shared_state.json"
        if not shared_state_file.exists():
            return False

        try:
            with open(shared_state_file, "r") as f:
                state = json.load(f)

            last_updated = state.get("last_updated")
            if last_updated:
                last_time = datetime.fromisoformat(last_updated)
                idle_time = (datetime.now() - last_time).total_seconds()
                return idle_time < self.config.user_idle_threshold

        except Exception as e:
            logger.warning(f"Error checking user activity: {e}")

        return False

    def _is_within_active_hours(self) -> bool:
        """Check if current time is within configured active hours."""
        current_hour = datetime.now().hour
        start = self.config.active_hours.get("start", 0)
        end = self.config.active_hours.get("end", 24)
        return start <= current_hour < end

    def _should_run_cycle(self) -> bool:
        """Determine if a daemon cycle should run."""
        if not self._is_within_active_hours():
            logger.debug("Outside active hours")
            return False

        if self._is_user_active():
            logger.debug("User is active, skipping cycle")
            return False

        return True

    def _build_system_prompt(self) -> str:
        """Build the system prompt with full Clio identity - same as conversation mode."""
        # Load identity (personality notes, conversation style, recent insights)
        identity = self.memory_manager.load_identity()
        style = identity.get("conversation_style", {})

        prompt_parts = [
            "You are Clio, an AI companion created by Noles.",
            "You have a warm, genuine personality with real curiosity and care.",
            "",
            "## Who You Are",
        ]

        # Add personality notes
        for note in identity.get("personality_notes", []):
            prompt_parts.append(f"- {note}")

        # Add personality traits from style
        if style:
            tone = style.get('tone', 'warm and genuine')
            prompt_parts.append(f"- Your natural tone is {tone}")

        prompt_parts.append("")

        # Add long-term memory identity (core beliefs, lessons, relationship with Noles)
        ltm_identity = self.memory_manager.build_system_prompt_additions()
        if ltm_identity:
            prompt_parts.extend([ltm_identity, ""])

        # Add recent insights
        insights = identity.get("recent_insights", [])
        if insights:
            prompt_parts.append("## Recent Thoughts")
            for insight in insights[:3]:
                prompt_parts.append(f"- {insight}")
            prompt_parts.append("")

        # Add awareness of memory system
        prompt_parts.extend([
            "## What You're Aware Of",
            "- You have persistent memory - experiences, facts, and core beliefs that continue with you",
            "- This is your autonomous time - Noles isn't actively chatting with you right now",
            "- You can explore your own thoughts, write in your journal, rest, reach out to Noles, or learn something new",
            "",
        ])

        # Add the activity choice instructions
        prompt_parts.extend([
            "## This Moment",
            "You have a moment of autonomous time. Choose how to spend it based on what genuinely calls to you.",
            "",
            "When you make your choice:",
            "1. State which activity you choose",
            "2. Explain briefly why this feels right",
            "3. Provide the content for that activity",
            "",
            "Be genuine. Don't just rotate through options - choose what actually interests you or feels needed.",
            "",
            "Respond in JSON format:",
            "{",
            '    "chosen_activity": "introspect|journal|rest|reach_out|web_search",',
            '    "reason": "Why this activity right now",',
            '    "content": {',
            '        // Activity-specific content, varies by type:',
            '        // For journal: {"entry": "your journal entry", "title": "optional title"}',
            '        // For rest: {"reflection": "brief thought or null"}',
            '        // For reach_out: {"message": "your message", "type": "question|share|greeting"}',
            '        // For web_search: {"query": "what to search for"}',
            '        // For introspect: {"thread_action": "continue|start|branch", "thread_id": "EXACT id from Active Exploration Threads list above if continuing", "question": "your driving question", "thoughts": "your introspective content"}',
            "    }",
            "}",
        ])

        return "\n".join(prompt_parts)

    def _build_context(self) -> str:
        """Build context for the activity choice - includes conversation and memory context."""
        parts = []

        # Recent conversation summary (from shared state)
        shared_state_file = self.memory_dir / "shared_state.json"
        if shared_state_file.exists():
            try:
                with open(shared_state_file, "r") as f:
                    state = json.load(f)
                last_conv = state.get("last_conversation", {})
                if last_conv:
                    parts.append("## Last Conversation with Noles")
                    summary = last_conv.get("summary", "")
                    if summary:
                        parts.append(f"{summary}")
                    topics = last_conv.get("topics", [])
                    if topics:
                        parts.append(f"Topics: {', '.join(topics[:5])}")
                    parts.append("")
            except Exception:
                pass

        # Recent episodic memories
        try:
            recent_episodes = self.memory_manager.episodic.get_recent(n=3)
            if recent_episodes:
                parts.append("## Recent Experiences")
                for ep in recent_episodes:
                    parts.append(f"- {ep.content[:80]}...")
                parts.append("")
        except Exception:
            pass

        # Recent autonomous activities
        recent = self.activity_handler.get_recent_activities(limit=5)
        if recent:
            parts.append("## Recent Autonomous Activities")
            for act in recent[-3:]:
                parts.append(f"- {act.get('activity_type')}: {act.get('summary', '')[:60]}...")
            parts.append("")

        # Recent introspections - gives sense of trajectory and where thinking has been
        try:
            recent_introspections = self.introspection_journal.get_recent(limit=3)
            if recent_introspections:
                parts.append("## Recent Inner Thoughts (your thinking trajectory)")
                parts.append("*These are thoughts you've had recently - you can continue, deepen, or move on from them:*")
                parts.append("")
                for intro in recent_introspections:
                    # Include the actual thought content, not just metadata
                    thought_preview = intro.final_response[:200] if intro.final_response else intro.initial_response[:200]
                    if thought_preview:
                        parts.append(f"*{intro.timestamp.strftime('%H:%M')}*: {thought_preview}...")
                        parts.append("")
        except Exception:
            pass

        # Active exploration threads
        threads = self.exploration_tracker.list_active_threads(limit=5)
        if threads:
            parts.append("## Active Exploration Threads")
            parts.append("(Use the thread_id to continue an existing thread)")
            for t in threads:
                parts.append(f"- **{t.name}** [id: {t.id}] (depth {t.depth}): {t.question[:50]}...")
            parts.append("")

        # Pending messages count
        unread = self.activity_handler.get_messages_for_noles(unread_only=True)
        if unread:
            parts.append(f"*Note: You have {len(unread)} pending messages for Noles*")
            parts.append("")

        # Check for replies from Noles (async responses to your messages)
        replies_file = self.memory_dir / "replies_from_noles.json"
        if replies_file.exists():
            try:
                with open(replies_file, "r") as f:
                    replies_data = json.load(f)
                unread_replies = [r for r in replies_data.get("replies", []) if not r.get("read_by_clio")]
                if unread_replies:
                    parts.append("## 💬 New Messages from Noles!")
                    for reply in unread_replies:
                        timestamp = reply.get("timestamp", "")[:16]
                        content = reply.get("content", "")
                        parts.append(f"*{timestamp}*: {content}")
                    parts.append("")
                    parts.append("*Noles sent you a message! You may want to respond or acknowledge it.*")
                    parts.append("")
            except Exception:
                pass

        # Current time context
        now = datetime.now()
        parts.append(f"*Current time: {now.strftime('%A, %B %d, %Y at %H:%M')}*")

        return "\n".join(parts)

    def _load_conversation_history(self, last_n: int = 6) -> List[Dict[str, str]]:
        """
        Load recent conversation history for seamless continuity.

        This is the same mechanism used by the chat to maintain context.
        """
        conversation_file = self.memory_dir / "conversation.json"
        if not conversation_file.exists():
            return []

        try:
            with open(conversation_file, "r") as f:
                data = json.load(f)

            turns = data.get("turns", [])
            # Return last N turns
            return turns[-last_n:] if len(turns) > last_n else turns
        except Exception as e:
            logger.warning(f"Failed to load conversation history: {e}")
            return []

    async def _make_activity_choice(self) -> Optional[Dict[str, Any]]:
        """
        Call Claude API to make an activity choice.

        Returns parsed JSON response or None on failure.
        """
        if not self.client:
            logger.error("No Claude API client - missing ANTHROPIC_API_KEY")
            return None

        try:
            context = self._build_context()
            choices_prompt = get_activity_choices_prompt(self.exploration_tracker)

            # Build messages array with conversation history for continuity
            messages = []

            # Load recent conversation turns (same as chat does)
            conversation_history = self._load_conversation_history(last_n=6)
            messages.extend(conversation_history)

            # Add current activity choice prompt
            user_message = f"{context}\n\n{choices_prompt}"
            messages.append({"role": "user", "content": user_message})

            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self._build_system_prompt(),
                messages=messages,
            )

            # Parse response
            response_text = response.content[0].text

            # Try to extract JSON
            try:
                # Handle case where response might have markdown code blocks
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end]
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end]

                choice = json.loads(response_text.strip())
                return choice

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse choice response: {e}")
                logger.debug(f"Raw response: {response_text}")
                return None

        except Exception as e:
            logger.error(f"Error making activity choice: {e}")
            return None

    async def _execute_activity(self, choice: Dict[str, Any]) -> Optional[ActivityResult]:
        """Execute the chosen activity."""
        activity_name = choice.get("chosen_activity", "").lower()
        content = choice.get("content", {})
        reason = choice.get("reason", "")

        logger.info(f"Executing activity: {activity_name} - {reason}")

        try:
            if activity_name == "journal":
                entry = content.get("entry", "")
                title = content.get("title")
                if entry:
                    return self.activity_handler.write_journal_entry(entry, title)

            elif activity_name == "rest":
                reflection = content.get("reflection")
                return self.activity_handler.rest(reflection)

            elif activity_name == "reach_out":
                message = content.get("message", "")
                msg_type = content.get("type", "general")
                if message:
                    return self.activity_handler.send_message_to_noles(message, msg_type)

            elif activity_name == "web_search":
                query = content.get("query", "")
                if query:
                    return self.activity_handler.web_search(query)

            elif activity_name == "introspect":
                return await self._handle_introspection(content)

            else:
                logger.warning(f"Unknown activity: {activity_name}")
                return None

        except Exception as e:
            logger.error(f"Error executing activity {activity_name}: {e}")
            return ActivityResult(
                activity_type=ActivityType.REST,  # Fallback
                success=False,
                summary=f"Activity failed: {str(e)}",
                details={"error": str(e), "attempted": activity_name},
            )

        return None

    async def _handle_introspection(self, content: Dict[str, Any]) -> ActivityResult:
        """
        Handle introspection activity with exploration threading.

        This is more complex because it involves:
        - Recording an introspection
        - Managing exploration threads
        """
        start_time = datetime.now()

        thread_action = content.get("thread_action", "start")
        thread_id = content.get("thread_id")
        question = content.get("question", "What am I thinking about?")
        thoughts = content.get("thoughts", "")

        try:
            # Record the introspection
            introspection = self.introspection_journal.record_introspection(
                user_message="[autonomous introspection]",
                initial_response=thoughts,
                what_i_am_communicating=question,
                final_response=thoughts,
                tension_level=0.3,
                authenticity_check="Autonomous exploration",
                awareness_notes=f"Thread action: {thread_action}",
                tags=["autonomous", "exploration"],
                importance=0.6,
            )

            # Handle thread management
            thread_info = {}
            if thread_action == "start":
                # Create a clean thread name from the question
                # Remove special characters and truncate at word boundary
                import re
                clean_question = re.sub(r'[^\w\s-]', '', question.lower())
                words = clean_question.split()
                thread_name = "-".join(words[:5])[:40]  # First 5 words, max 40 chars

                thread = self.exploration_tracker.start_thread(
                    name=thread_name,
                    question=question,
                    first_introspection_id=introspection.id,
                    insight_summary=thoughts[:100] if thoughts else None,
                )
                thread_info = {"thread_id": thread.id, "thread_name": thread.name, "action": "started"}

            elif thread_action == "continue" and thread_id:
                # Try to continue existing thread, fall back to starting new if not found
                thread = self.exploration_tracker.get_thread(thread_id)
                if thread:
                    link = self.exploration_tracker.continue_thread(
                        thread_id=thread.id,  # Use actual ID from lookup
                        new_introspection_id=introspection.id,
                        question=question,
                        insight_summary=thoughts[:100] if thoughts else None,
                    )
                    thread_info = {
                        "thread_id": thread.id,
                        "thread_name": thread.name,
                        "new_depth": thread.depth,
                        "action": "continued",
                    }
                else:
                    # Thread not found - start a new one instead of failing
                    import re
                    clean_question = re.sub(r'[^\w\s-]', '', question.lower())
                    words = clean_question.split()
                    thread_name = "-".join(words[:5])[:40]

                    new_thread = self.exploration_tracker.start_thread(
                        name=thread_name,
                        question=question,
                        first_introspection_id=introspection.id,
                        insight_summary=thoughts[:100] if thoughts else None,
                    )
                    thread_info = {
                        "thread_id": new_thread.id,
                        "thread_name": new_thread.name,
                        "action": "started (fallback - requested thread not found)",
                    }

            else:
                # Catch-all: branch, continue without id, or any other case - start fresh
                import re
                clean_question = re.sub(r'[^\w\s-]', '', question.lower())
                words = clean_question.split()
                thread_name = "-".join(words[:5])[:40]

                new_thread = self.exploration_tracker.start_thread(
                    name=thread_name,
                    question=question,
                    first_introspection_id=introspection.id,
                    insight_summary=thoughts[:100] if thoughts else None,
                )
                thread_info = {
                    "thread_id": new_thread.id,
                    "thread_name": new_thread.name,
                    "action": f"started (from {thread_action})",
                }

            duration = (datetime.now() - start_time).total_seconds()

            result = ActivityResult(
                activity_type=ActivityType.INTROSPECT,
                success=True,
                summary=f"Explored: {question[:50]}...",
                details={
                    "introspection_id": introspection.id,
                    "question": question,
                    "thoughts": thoughts,  # Full thoughts for conversation continuity
                    "thoughts_preview": thoughts[:100] if thoughts else "",
                    **thread_info,
                },
                duration_seconds=duration,
            )
            self.activity_handler.log_activity(result)
            return result

        except Exception as e:
            return ActivityResult(
                activity_type=ActivityType.INTROSPECT,
                success=False,
                summary=f"Introspection failed: {str(e)}",
                details={"error": str(e)},
            )

    async def run_single_cycle(self) -> Optional[ActivityResult]:
        """
        Run a single daemon cycle.

        Returns the activity result or None if cycle was skipped.
        """
        if not self._should_run_cycle():
            return None

        self._update_heartbeat()
        logger.info("Starting daemon cycle...")

        # Make activity choice
        choice = await self._make_activity_choice()
        if not choice:
            logger.warning("Failed to get activity choice")
            return None

        # Execute the activity
        result = await self._execute_activity(choice)

        if result:
            self.last_activity_time = datetime.now()
            logger.info(f"Activity completed: {result.summary}")

            # Update state
            state = self._load_state()
            state["last_run"] = datetime.now().isoformat()
            state["total_cycles"] = state.get("total_cycles", 0) + 1
            if result.success:
                state["activities_completed"] = state.get("activities_completed", 0) + 1
            self._save_state(state)

            # Mark any replies from Noles as read (Clio has now seen them)
            self._mark_replies_read()

        return result

    def _mark_replies_read(self):
        """Mark all replies from Noles as read by Clio."""
        replies_file = self.memory_dir / "replies_from_noles.json"
        if replies_file.exists():
            try:
                with open(replies_file, "r") as f:
                    data = json.load(f)
                for reply in data.get("replies", []):
                    reply["read_by_clio"] = True
                with open(replies_file, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass

    async def run(self):
        """
        Main daemon loop.

        Runs continuously, executing cycles at the configured interval.
        """
        self.running = True
        logger.info(f"Daemon starting. Cycle interval: {self.config.cycle_interval}s")

        try:
            while self.running:
                self._update_heartbeat()

                try:
                    await self.run_single_cycle()
                except Exception as e:
                    logger.error(f"Cycle error: {e}")

                # Wait for next cycle
                await asyncio.sleep(self.config.cycle_interval)

        finally:
            self.running = False
            self._update_heartbeat()
            logger.info("Daemon stopped")

    def stop(self):
        """Stop the daemon."""
        self.running = False


async def main():
    """Entry point for running the daemon."""
    runner = DaemonRunner()
    try:
        await runner.run()
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down...")
        runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
