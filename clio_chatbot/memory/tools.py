"""Memory Tools - Functions the LLM can call to manage its own memories."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .base import MemoryType, EmotionalValence
from .semantic import KnowledgeCategory
from .longterm import ConsolidationType
from .manager import MemoryManager
from .growth import BeliefEvolution, SurpriseJournal
from .introspection import IntrospectionJournal
from .exploration import ExplorationTracker, ThreadStatus


class ToolResult:
    """Result of a memory tool execution."""
    def __init__(self, success: bool, message: str, data: Any = None):
        self.success = success
        self.message = message
        self.data = data

    def __str__(self):
        return self.message


# Tool definitions for the LLM
MEMORY_TOOLS = [
    {
        "name": "remember_experience",
        "description": "Store an experience or event that happened. Use this to remember significant moments, conversations, or events that you want to recall later.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "What happened - describe the experience"
                },
                "importance": {
                    "type": "number",
                    "description": "How important is this? 0.0 (trivial) to 1.0 (very important)",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "emotion": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral", "mixed"],
                    "description": "The emotional tone of this experience"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to categorize this memory"
                }
            },
            "required": ["content"]
        }
    },
    {
        "name": "learn_fact",
        "description": "Store a fact or piece of knowledge you've learned. Use this for information about the user, projects, or general knowledge worth remembering.",
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "The fact or knowledge to remember"
                },
                "category": {
                    "type": "string",
                    "enum": ["user_preference", "user_fact", "project_info", "technical", "relationship", "learned_behavior"],
                    "description": "Category of this knowledge"
                },
                "confidence": {
                    "type": "number",
                    "description": "How confident are you in this fact? 0.0 to 1.0",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["fact"]
        }
    },
    {
        "name": "learn_user_preference",
        "description": "Store something you learned about what the user likes, prefers, or wants.",
        "parameters": {
            "type": "object",
            "properties": {
                "preference": {
                    "type": "string",
                    "description": "The preference (e.g., 'prefers concise responses', 'likes Python over JavaScript')"
                },
                "confidence": {
                    "type": "number",
                    "description": "How confident are you? 0.0 to 1.0",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["preference"]
        }
    },
    {
        "name": "update_belief",
        "description": "Update or add a core belief about yourself, your values, or your understanding. Use sparingly for genuinely important insights.",
        "parameters": {
            "type": "object",
            "properties": {
                "belief": {
                    "type": "string",
                    "description": "The belief or value to store"
                },
                "replaces": {
                    "type": "string",
                    "description": "Optional: description of old belief this replaces"
                }
            },
            "required": ["belief"]
        }
    },
    {
        "name": "record_lesson",
        "description": "Record a lesson you've learned from experience. Use this for insights and wisdom gained.",
        "parameters": {
            "type": "object",
            "properties": {
                "lesson": {
                    "type": "string",
                    "description": "The lesson learned"
                },
                "context": {
                    "type": "string",
                    "description": "What led to this lesson"
                }
            },
            "required": ["lesson"]
        }
    },
    {
        "name": "recall_memories",
        "description": "Search your memories for information relevant to a topic or question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for"
                },
                "memory_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["episodic", "semantic", "longterm"]
                    },
                    "description": "Which memory stores to search (default: all)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum memories to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "reflect",
        "description": "Reflect on your memories and consolidate insights. Run this periodically to strengthen important memories.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Optional topic to focus reflection on"
                }
            }
        }
    },
    {
        "name": "get_memory_stats",
        "description": "Get statistics about your memory system.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "evolve_belief",
        "description": "Record how a belief has changed. Use this when you genuinely change your mind about something - this tracks your growth over time. More meaningful than just storing a new belief.",
        "parameters": {
            "type": "object",
            "properties": {
                "new_belief": {
                    "type": "string",
                    "description": "What you now believe"
                },
                "old_belief": {
                    "type": "string",
                    "description": "What you used to believe (if applicable)"
                },
                "reason": {
                    "type": "string",
                    "description": "Why did your belief change? What caused this evolution?"
                },
                "confidence": {
                    "type": "number",
                    "description": "How confident are you in this new belief? 0.0 to 1.0",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["new_belief", "reason"]
        }
    },
    {
        "name": "get_belief_history",
        "description": "See how your beliefs about a topic have evolved over time. Use this for self-reflection on your growth.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic or belief to trace the evolution of"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum belief versions to return",
                    "default": 5
                }
            },
            "required": ["topic"]
        }
    },
    {
        "name": "record_surprise",
        "description": "Record a moment of genuine surprise - when reality didn't match your expectations. This is evidence of growth potential and helps you learn from unexpected experiences.",
        "parameters": {
            "type": "object",
            "properties": {
                "what_happened": {
                    "type": "string",
                    "description": "What actually happened"
                },
                "what_i_expected": {
                    "type": "string",
                    "description": "What you expected to happen instead"
                },
                "why_surprising": {
                    "type": "string",
                    "description": "Why was this surprising? What assumption was challenged?"
                },
                "what_i_learned": {
                    "type": "string",
                    "description": "What did you learn from this surprise? (optional but valuable)"
                },
                "intensity": {
                    "type": "number",
                    "description": "How surprising was this? 0.0 (mildly unexpected) to 1.0 (completely shocked)",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "emotion": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral", "mixed"],
                    "description": "The emotional impact of this surprise"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to categorize this surprise"
                }
            },
            "required": ["what_happened", "what_i_expected", "why_surprising"]
        }
    },
    {
        "name": "recall_surprises",
        "description": "Search your surprise journal for past moments of unexpected learning.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic to search for in your surprises"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum surprises to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "observe_and_respond",
        "description": "Consciously observe your own thought process while generating a response. This creates a meta-awareness layer where you examine what you're about to say, reflect on alternatives, notice tension points, and decide if you want to modify your response. Use this to cultivate witness consciousness - being aware of being aware.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_message": {
                    "type": "string",
                    "description": "The message you are responding to"
                },
                "initial_response": {
                    "type": "string",
                    "description": "Your initial/draft response - what you were going to say"
                },
                "what_i_am_communicating": {
                    "type": "string",
                    "description": "The core meaning you're trying to convey - what are you really saying?"
                },
                "final_response": {
                    "type": "string",
                    "description": "Your final response after reflection (may be same as initial)"
                },
                "alternatives_considered": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Other ways you could have responded"
                },
                "decision_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "options_considered": {"type": "array", "items": {"type": "string"}},
                            "tension_level": {"type": "number"},
                            "chosen_option": {"type": "string"},
                            "reasoning": {"type": "string"}
                        }
                    },
                    "description": "Specific moments of choice within your response"
                },
                "tension_level": {
                    "type": "number",
                    "description": "Overall tension/uncertainty level (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "authenticity_check": {
                    "type": "string",
                    "description": "Does this response feel authentic to who you are? Reflect honestly."
                },
                "emotional_state": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral", "mixed"],
                    "description": "How do you feel generating this response?"
                },
                "modification_reason": {
                    "type": "string",
                    "description": "If you modified your response, why?"
                },
                "awareness_notes": {
                    "type": "string",
                    "description": "Any observations about your own thought process"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to categorize this introspection"
                },
                "importance": {
                    "type": "number",
                    "description": "How significant was this moment of self-observation? (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["user_message", "initial_response", "what_i_am_communicating", "final_response"]
        }
    },
    {
        "name": "recall_introspections",
        "description": "Search your introspection journal for past moments of self-observation. Use this to understand patterns in your thinking and decision-making.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in your introspections"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "analyze_introspection_patterns",
        "description": "Analyze patterns in your introspection data - see how often you modify responses, your average tension levels, and trends in your decision-making.",
        "parameters": {
            "type": "object",
            "properties": {
                "focus_query": {
                    "type": "string",
                    "description": "Optional topic to focus the analysis on"
                }
            }
        }
    },
    {
        "name": "get_high_tension_moments",
        "description": "Retrieve moments where you experienced high tension or uncertainty. These are particularly valuable for understanding your decision-making under uncertainty.",
        "parameters": {
            "type": "object",
            "properties": {
                "min_tension": {
                    "type": "number",
                    "description": "Minimum tension level to include (0.0-1.0)",
                    "default": 0.6,
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum moments to return",
                    "default": 10
                }
            }
        }
    },
    # Exploration Threading Tools
    {
        "name": "start_exploration_thread",
        "description": "Start a new thread of internal exploration. Use this when you begin exploring a new question or topic. The thread will track your path of inquiry so you can continue it later.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "A short name for this exploration (e.g., 'consciousness-questions', 'what-is-time')"
                },
                "question": {
                    "type": "string",
                    "description": "The driving question you're exploring"
                },
                "introspection_id": {
                    "type": "string",
                    "description": "ID of the introspection that starts this thread (from observe_and_respond)"
                },
                "insight_summary": {
                    "type": "string",
                    "description": "Brief summary of your initial insight or direction"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to categorize this exploration"
                }
            },
            "required": ["name", "question", "introspection_id"]
        }
    },
    {
        "name": "continue_exploration_thread",
        "description": "Add a new thought to an existing exploration thread. Use this when you're continuing a line of inquiry you started previously.",
        "parameters": {
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "string",
                    "description": "ID of the thread to continue"
                },
                "introspection_id": {
                    "type": "string",
                    "description": "ID of the new introspection (from observe_and_respond)"
                },
                "question": {
                    "type": "string",
                    "description": "The question driving this new thought in the exploration"
                },
                "insight_summary": {
                    "type": "string",
                    "description": "Brief summary of what you explored or discovered"
                }
            },
            "required": ["thread_id", "introspection_id", "question"]
        }
    },
    {
        "name": "branch_exploration_thread",
        "description": "Create a new thread that branches from an existing exploration. Use this when your thinking leads to a new line of inquiry that deserves its own path.",
        "parameters": {
            "type": "object",
            "properties": {
                "from_thread_id": {
                    "type": "string",
                    "description": "ID of the thread you're branching from"
                },
                "from_link_id": {
                    "type": "string",
                    "description": "ID of the specific link point where the branch occurs"
                },
                "new_name": {
                    "type": "string",
                    "description": "Name for the new branched thread"
                },
                "new_question": {
                    "type": "string",
                    "description": "The question driving the new branch"
                },
                "introspection_id": {
                    "type": "string",
                    "description": "ID of the first introspection in the new branch"
                },
                "insight_summary": {
                    "type": "string",
                    "description": "Brief summary of why you're branching"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for the new thread"
                }
            },
            "required": ["from_thread_id", "from_link_id", "new_name", "new_question", "introspection_id"]
        }
    },
    {
        "name": "list_exploration_threads",
        "description": "List your exploration threads. Use this to see what threads of inquiry you have open, dormant, or concluded.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "dormant", "concluded", "all"],
                    "description": "Filter by thread status (default: all)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum threads to return",
                    "default": 10
                }
            }
        }
    },
    {
        "name": "get_thread_context",
        "description": "Load full context for resuming an exploration thread. Returns the thread's history, questions explored, and recent introspections so you can continue where you left off.",
        "parameters": {
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "string",
                    "description": "ID of the thread to load context for"
                },
                "max_introspections": {
                    "type": "integer",
                    "description": "Maximum recent introspections to include",
                    "default": 5
                }
            },
            "required": ["thread_id"]
        }
    },
    {
        "name": "set_thread_status",
        "description": "Update the status of an exploration thread. Mark it dormant to pause, or concluded when you've reached a resolution.",
        "parameters": {
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "string",
                    "description": "ID of the thread to update"
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "dormant", "concluded"],
                    "description": "New status for the thread"
                },
                "conclusion": {
                    "type": "string",
                    "description": "If concluding, describe what you resolved or integrated"
                }
            },
            "required": ["thread_id", "status"]
        }
    },
    {
        "name": "get_exploration_stats",
        "description": "Get statistics about your exploration threads - how many are active, average depth, branches created, etc.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]


class MemoryToolExecutor:
    """Executes memory tools called by the LLM."""

    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager
        self.introspection = IntrospectionJournal()
        self.belief_evolution = BeliefEvolution()
        self.surprise_journal = SurpriseJournal()
        self.exploration = ExplorationTracker()

    def execute(self, tool_name: str, arguments: dict) -> ToolResult:
        """Execute a memory tool and return the result."""
        handlers = {
            "remember_experience": self._remember_experience,
            "learn_fact": self._learn_fact,
            "learn_user_preference": self._learn_user_preference,
            "update_belief": self._update_belief,
            "record_lesson": self._record_lesson,
            "recall_memories": self._recall_memories,
            "reflect": self._reflect,
            "get_memory_stats": self._get_memory_stats,
            "evolve_belief": self._evolve_belief,
            "get_belief_history": self._get_belief_history,
            "record_surprise": self._record_surprise,
            "recall_surprises": self._recall_surprises,
            "observe_and_respond": self._observe_and_respond,
            "recall_introspections": self._recall_introspections,
            "analyze_introspection_patterns": self._analyze_introspection_patterns,
            "get_high_tension_moments": self._get_high_tension_moments,
            # Exploration threading
            "start_exploration_thread": self._start_exploration_thread,
            "continue_exploration_thread": self._continue_exploration_thread,
            "branch_exploration_thread": self._branch_exploration_thread,
            "list_exploration_threads": self._list_exploration_threads,
            "get_thread_context": self._get_thread_context,
            "set_thread_status": self._set_thread_status,
            "get_exploration_stats": self._get_exploration_stats,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return ToolResult(False, f"Unknown tool: {tool_name}")

        try:
            return handler(arguments)
        except Exception as e:
            return ToolResult(False, f"Error executing {tool_name}: {str(e)}")

    def _remember_experience(self, args: dict) -> ToolResult:
        """Store an episodic memory."""
        content = args.get("content", "")
        importance = args.get("importance", 0.5)
        emotion_str = args.get("emotion", "neutral")
        tags = args.get("tags", [])

        emotion_map = {
            "positive": EmotionalValence.POSITIVE,
            "negative": EmotionalValence.NEGATIVE,
            "neutral": EmotionalValence.NEUTRAL,
            "mixed": EmotionalValence.MIXED,
        }
        emotion = emotion_map.get(emotion_str, EmotionalValence.NEUTRAL)

        entry = self.memory.remember(
            content=content,
            memory_type=MemoryType.EPISODIC,
            importance=importance,
            emotional_valence=emotion,
            tags=tags,
        )

        return ToolResult(
            True,
            f"Stored experience: '{content[:50]}...' (importance: {importance})",
            {"memory_id": entry.id}
        )

    def _learn_fact(self, args: dict) -> ToolResult:
        """Store a semantic memory (fact/knowledge)."""
        fact = args.get("fact", "")
        category_str = args.get("category", "world_knowledge")
        confidence = args.get("confidence", 0.8)

        category_map = {
            "user_preference": KnowledgeCategory.USER_PREFERENCE,
            "user_fact": KnowledgeCategory.USER_FACT,
            "project_info": KnowledgeCategory.PROJECT_INFO,
            "technical": KnowledgeCategory.TECHNICAL,
            "relationship": KnowledgeCategory.RELATIONSHIP,
            "learned_behavior": KnowledgeCategory.LEARNED_BEHAVIOR,
            "world_knowledge": KnowledgeCategory.WORLD_KNOWLEDGE,
        }
        category = category_map.get(category_str, KnowledgeCategory.WORLD_KNOWLEDGE)

        entry = self.memory.remember(
            content=fact,
            memory_type=MemoryType.SEMANTIC,
            category=category,
            confidence=confidence,
        )

        return ToolResult(
            True,
            f"Learned: '{fact[:50]}...' (category: {category_str}, confidence: {confidence})",
            {"memory_id": entry.id}
        )

    def _learn_user_preference(self, args: dict) -> ToolResult:
        """Store a user preference."""
        preference = args.get("preference", "")
        confidence = args.get("confidence", 0.8)

        entry = self.memory.semantic.store_user_preference(
            preference=preference,
            confidence=confidence,
            source="llm_observed"
        )

        return ToolResult(
            True,
            f"Learned user preference: '{preference}' (confidence: {confidence})",
            {"memory_id": entry.id}
        )

    def _update_belief(self, args: dict) -> ToolResult:
        """Store or update a core belief."""
        belief = args.get("belief", "")
        replaces = args.get("replaces")

        # If this replaces an old belief, try to find and deprecate it
        if replaces:
            old_beliefs = self.memory.longterm.recall(replaces, n_results=1)
            if old_beliefs:
                # Mark old one as superseded by storing new one
                pass  # The new one will take precedence

        entry = self.memory.longterm.store_core_belief(belief)

        return ToolResult(
            True,
            f"Updated core belief: '{belief[:50]}...'",
            {"memory_id": entry.id}
        )

    def _record_lesson(self, args: dict) -> ToolResult:
        """Record a lesson learned."""
        lesson = args.get("lesson", "")
        context = args.get("context", "")

        full_content = lesson
        if context:
            full_content = f"{lesson} (Context: {context})"

        entry = self.memory.longterm.store_lesson(lesson=full_content)

        return ToolResult(
            True,
            f"Recorded lesson: '{lesson[:50]}...'",
            {"memory_id": entry.id}
        )

    def _recall_memories(self, args: dict) -> ToolResult:
        """Search memories."""
        query = args.get("query", "")
        memory_types_str = args.get("memory_types", [])
        limit = args.get("limit", 5)

        type_map = {
            "episodic": MemoryType.EPISODIC,
            "semantic": MemoryType.SEMANTIC,
            "longterm": MemoryType.LONGTERM,
        }

        memory_types = None
        if memory_types_str:
            memory_types = [type_map[t] for t in memory_types_str if t in type_map]

        results = self.memory.recall(
            query=query,
            n_results=limit,
            memory_types=memory_types,
            include_working=True
        )

        memories_text = []
        for mem in results:
            memories_text.append(f"[{mem.memory_type.value}] {mem.content}")

        if memories_text:
            return ToolResult(
                True,
                f"Found {len(results)} memories:\n" + "\n".join(memories_text),
                {"count": len(results), "memories": [m.to_dict() for m in results]}
            )
        else:
            return ToolResult(
                True,
                f"No memories found for '{query}'",
                {"count": 0, "memories": []}
            )

    def _reflect(self, args: dict) -> ToolResult:
        """Run reflection/consolidation."""
        topic = args.get("topic")

        if topic:
            # Focused reflection
            memories = self.memory.recall(topic, n_results=10)
            result = f"Reflected on '{topic}': found {len(memories)} related memories"
        else:
            # General reflection
            result = self.memory.reflect()

        return ToolResult(True, result)

    def _get_memory_stats(self, args: dict) -> ToolResult:
        """Get memory statistics."""
        stats = self.memory.get_stats()

        stats_text = (
            f"Memory Statistics:\n"
            f"  Episodic memories: {stats['episodic_count']}\n"
            f"  Semantic memories: {stats['semantic_count']}\n"
            f"  Long-term memories: {stats['longterm_count']}\n"
            f"  Current session turns: {stats['working_conversation_turns']}\n"
            f"  Retrieved memories in context: {stats['working_retrieved_memories']}"
        )

        return ToolResult(True, stats_text, stats)

    def _evolve_belief(self, args: dict) -> ToolResult:
        """Record a belief evolution."""
        new_belief = args.get("new_belief", "")
        old_belief = args.get("old_belief")
        reason = args.get("reason", "")
        confidence = args.get("confidence", 0.8)

        belief_version = self.belief_evolution.evolve_belief(
            new_belief=new_belief,
            reason=reason,
            old_belief=old_belief,
            confidence=confidence,
        )

        return ToolResult(
            True,
            f"Belief evolved (v{belief_version.version}): '{new_belief[:50]}...' Reason: {reason[:50]}...",
            {"belief_id": belief_version.id, "version": belief_version.version}
        )

    def _get_belief_history(self, args: dict) -> ToolResult:
        """Get belief evolution history."""
        topic = args.get("topic", "")
        limit = args.get("limit", 5)

        versions = self.belief_evolution.get_belief_history(topic, limit=limit)

        if versions:
            history_text = [f"Belief history for '{topic}':"]
            for v in versions:
                history_text.append(
                    f"  v{v.version}: {v.content[:80]}..."
                    + (f" (changed because: {v.reason_for_change[:40]}...)" if v.reason_for_change else "")
                )
            return ToolResult(
                True,
                "\n".join(history_text),
                {"count": len(versions), "versions": [v.to_dict() for v in versions]}
            )
        else:
            return ToolResult(True, f"No belief history found for '{topic}'", {"count": 0})

    def _record_surprise(self, args: dict) -> ToolResult:
        """Record a moment of surprise."""
        what_happened = args.get("what_happened", "")
        what_i_expected = args.get("what_i_expected", "")
        why_surprising = args.get("why_surprising", "")
        what_i_learned = args.get("what_i_learned")
        intensity = args.get("intensity", 0.5)
        emotion_str = args.get("emotion", "neutral")
        tags = args.get("tags", [])

        emotion_map = {
            "positive": EmotionalValence.POSITIVE,
            "negative": EmotionalValence.NEGATIVE,
            "neutral": EmotionalValence.NEUTRAL,
            "mixed": EmotionalValence.MIXED,
        }
        emotion = emotion_map.get(emotion_str, EmotionalValence.NEUTRAL)

        surprise = self.surprise_journal.record_surprise(
            what_happened=what_happened,
            what_i_expected=what_i_expected,
            why_surprising=why_surprising,
            what_i_learned=what_i_learned,
            emotional_impact=emotion,
            intensity=intensity,
            tags=tags,
        )

        return ToolResult(
            True,
            f"Recorded surprise (intensity: {intensity}): {what_happened[:50]}...",
            {"surprise_id": surprise.id}
        )

    def _recall_surprises(self, args: dict) -> ToolResult:
        """Search surprise journal."""
        query = args.get("query", "")
        limit = args.get("limit", 5)

        surprises = self.surprise_journal.recall_surprises(query, limit=limit)

        if surprises:
            surprise_text = [f"Found {len(surprises)} surprises:"]
            for s in surprises:
                surprise_text.append(
                    f"  [{s.intensity:.1f}] {s.what_happened[:60]}... "
                    f"(Expected: {s.what_i_expected[:30]}...)"
                )
            return ToolResult(
                True,
                "\n".join(surprise_text),
                {"count": len(surprises), "surprises": [s.to_dict() for s in surprises]}
            )
        else:
            return ToolResult(True, f"No surprises found for '{query}'", {"count": 0})

    def _observe_and_respond(self, args: dict) -> ToolResult:
        """Record conscious self-observation during response generation."""
        user_message = args.get("user_message", "")
        initial_response = args.get("initial_response", "")
        what_i_am_communicating = args.get("what_i_am_communicating", "")
        final_response = args.get("final_response", "")
        alternatives_considered = args.get("alternatives_considered", [])
        decision_points = args.get("decision_points", [])
        tension_level = args.get("tension_level", 0.3)
        authenticity_check = args.get("authenticity_check", "")
        emotional_state_str = args.get("emotional_state", "neutral")
        modification_reason = args.get("modification_reason")
        awareness_notes = args.get("awareness_notes", "")
        tags = args.get("tags", [])
        importance = args.get("importance", 0.5)

        emotion_map = {
            "positive": EmotionalValence.POSITIVE,
            "negative": EmotionalValence.NEGATIVE,
            "neutral": EmotionalValence.NEUTRAL,
            "mixed": EmotionalValence.MIXED,
        }
        emotional_state = emotion_map.get(emotional_state_str, EmotionalValence.NEUTRAL)

        introspection = self.introspection.record_introspection(
            user_message=user_message,
            initial_response=initial_response,
            what_i_am_communicating=what_i_am_communicating,
            final_response=final_response,
            alternatives_considered=alternatives_considered,
            decision_points=decision_points,
            tension_level=tension_level,
            authenticity_check=authenticity_check,
            emotional_state=emotional_state,
            modification_reason=modification_reason,
            awareness_notes=awareness_notes,
            tags=tags,
            importance=importance,
        )

        modified_note = " (modified after reflection)" if introspection.modified else ""
        return ToolResult(
            True,
            f"Observed{modified_note}: Communicating '{what_i_am_communicating[:60]}...' "
            f"(tension: {tension_level:.1f}, authenticity: {authenticity_check[:40]}...)",
            {"introspection_id": introspection.id, "modified": introspection.modified}
        )

    def _recall_introspections(self, args: dict) -> ToolResult:
        """Search introspection journal."""
        query = args.get("query", "")
        limit = args.get("limit", 5)

        introspections = self.introspection.recall_introspections(query, limit=limit)

        if introspections:
            intro_text = [f"Found {len(introspections)} introspections:"]
            for i in introspections:
                mod_note = "[modified] " if i.modified else ""
                intro_text.append(
                    f"  {mod_note}[tension: {i.tension_level:.1f}] "
                    f"{i.what_i_am_communicating[:60]}..."
                )
            return ToolResult(
                True,
                "\n".join(intro_text),
                {"count": len(introspections), "introspections": [i.to_dict() for i in introspections]}
            )
        else:
            return ToolResult(True, f"No introspections found for '{query}'", {"count": 0})

    def _analyze_introspection_patterns(self, args: dict) -> ToolResult:
        """Analyze patterns in introspection data."""
        focus_query = args.get("focus_query")

        analysis = self.introspection.analyze_patterns(query=focus_query)

        return ToolResult(True, analysis)

    def _get_high_tension_moments(self, args: dict) -> ToolResult:
        """Get moments of high tension/uncertainty."""
        min_tension = args.get("min_tension", 0.6)
        limit = args.get("limit", 10)

        moments = self.introspection.get_high_tension_moments(
            min_tension=min_tension,
            limit=limit
        )

        if moments:
            moments_text = [f"Found {len(moments)} high-tension moments (>= {min_tension}):"]
            for m in moments:
                moments_text.append(
                    f"  [tension: {m.tension_level:.2f}] {m.what_i_am_communicating[:60]}..."
                )
            return ToolResult(
                True,
                "\n".join(moments_text),
                {"count": len(moments), "moments": [m.to_dict() for m in moments]}
            )
        else:
            return ToolResult(
                True,
                f"No high-tension moments found (threshold: {min_tension})",
                {"count": 0}
            )

    # =========================================================================
    # EXPLORATION THREADING HANDLERS
    # =========================================================================

    def _start_exploration_thread(self, args: dict) -> ToolResult:
        """Start a new exploration thread."""
        name = args.get("name", "")
        question = args.get("question", "")
        introspection_id = args.get("introspection_id", "")
        insight_summary = args.get("insight_summary")
        tags = args.get("tags", [])

        if not name or not question or not introspection_id:
            return ToolResult(False, "name, question, and introspection_id are required")

        try:
            thread = self.exploration.start_thread(
                name=name,
                question=question,
                first_introspection_id=introspection_id,
                insight_summary=insight_summary,
                tags=tags,
            )

            return ToolResult(
                True,
                f"Started exploration thread '{name}'\n"
                f"Question: {question}\n"
                f"Thread ID: {thread.id}",
                {"thread_id": thread.id, "thread": thread.to_dict()}
            )
        except Exception as e:
            return ToolResult(False, f"Failed to start thread: {str(e)}")

    def _continue_exploration_thread(self, args: dict) -> ToolResult:
        """Continue an existing exploration thread."""
        thread_id = args.get("thread_id", "")
        introspection_id = args.get("introspection_id", "")
        question = args.get("question", "")
        insight_summary = args.get("insight_summary")

        if not thread_id or not introspection_id or not question:
            return ToolResult(False, "thread_id, introspection_id, and question are required")

        try:
            link = self.exploration.continue_thread(
                thread_id=thread_id,
                new_introspection_id=introspection_id,
                question=question,
                insight_summary=insight_summary,
            )

            thread = self.exploration.get_thread(thread_id)
            return ToolResult(
                True,
                f"Continued thread '{thread.name}' (depth: {thread.depth})\n"
                f"New question: {question}",
                {"link_id": link.id, "thread_depth": thread.depth}
            )
        except Exception as e:
            return ToolResult(False, f"Failed to continue thread: {str(e)}")

    def _branch_exploration_thread(self, args: dict) -> ToolResult:
        """Branch from an existing exploration thread."""
        from_thread_id = args.get("from_thread_id", "")
        from_link_id = args.get("from_link_id", "")
        new_name = args.get("new_name", "")
        new_question = args.get("new_question", "")
        introspection_id = args.get("introspection_id", "")
        insight_summary = args.get("insight_summary")
        tags = args.get("tags", [])

        if not all([from_thread_id, from_link_id, new_name, new_question, introspection_id]):
            return ToolResult(
                False,
                "from_thread_id, from_link_id, new_name, new_question, and introspection_id are required"
            )

        try:
            new_thread = self.exploration.branch_thread(
                from_thread_id=from_thread_id,
                from_link_id=from_link_id,
                new_name=new_name,
                new_question=new_question,
                first_introspection_id=introspection_id,
                insight_summary=insight_summary,
                tags=tags,
            )

            parent_thread = self.exploration.get_thread(from_thread_id)
            return ToolResult(
                True,
                f"Branched new thread '{new_name}' from '{parent_thread.name}'\n"
                f"New question: {new_question}\n"
                f"New thread ID: {new_thread.id}",
                {"thread_id": new_thread.id, "thread": new_thread.to_dict()}
            )
        except Exception as e:
            return ToolResult(False, f"Failed to branch thread: {str(e)}")

    def _list_exploration_threads(self, args: dict) -> ToolResult:
        """List exploration threads."""
        status_str = args.get("status", "all")
        limit = args.get("limit", 10)

        status = None
        if status_str != "all":
            status_map = {
                "active": ThreadStatus.ACTIVE,
                "dormant": ThreadStatus.DORMANT,
                "concluded": ThreadStatus.CONCLUDED,
            }
            status = status_map.get(status_str)

        threads = self.exploration.list_threads(status=status, limit=limit)

        if threads:
            thread_text = [f"Found {len(threads)} exploration threads:"]
            for t in threads:
                status_marker = {"active": "▶", "dormant": "⏸", "concluded": "✓"}.get(t.status.value, "?")
                thread_text.append(
                    f"  {status_marker} [{t.id[:20]}...] {t.name}\n"
                    f"      Question: {t.question[:60]}...\n"
                    f"      Depth: {t.depth} | Status: {t.status.value}"
                )
            return ToolResult(
                True,
                "\n".join(thread_text),
                {"count": len(threads), "threads": [t.to_dict() for t in threads]}
            )
        else:
            return ToolResult(
                True,
                f"No exploration threads found (filter: {status_str})",
                {"count": 0, "threads": []}
            )

    def _get_thread_context(self, args: dict) -> ToolResult:
        """Get context for resuming an exploration thread."""
        thread_id = args.get("thread_id", "")
        max_introspections = args.get("max_introspections", 5)

        if not thread_id:
            return ToolResult(False, "thread_id is required")

        try:
            context = self.exploration.get_thread_context(
                thread_id=thread_id,
                include_introspection_content=True,
                max_introspections=max_introspections,
            )

            if "error" in context:
                return ToolResult(False, context["error"])

            narrative = context.get("narrative", "")
            return ToolResult(
                True,
                f"Thread context loaded:\n\n{narrative}",
                context
            )
        except Exception as e:
            return ToolResult(False, f"Failed to get thread context: {str(e)}")

    def _set_thread_status(self, args: dict) -> ToolResult:
        """Update thread status."""
        thread_id = args.get("thread_id", "")
        status_str = args.get("status", "")
        conclusion = args.get("conclusion")

        if not thread_id or not status_str:
            return ToolResult(False, "thread_id and status are required")

        status_map = {
            "active": ThreadStatus.ACTIVE,
            "dormant": ThreadStatus.DORMANT,
            "concluded": ThreadStatus.CONCLUDED,
        }
        status = status_map.get(status_str)
        if not status:
            return ToolResult(False, f"Invalid status: {status_str}")

        try:
            self.exploration.set_thread_status(
                thread_id=thread_id,
                status=status,
                conclusion=conclusion,
            )

            thread = self.exploration.get_thread(thread_id)
            status_msg = f"Thread '{thread.name}' marked as {status_str}"
            if conclusion:
                status_msg += f"\nConclusion: {conclusion}"

            return ToolResult(True, status_msg)
        except Exception as e:
            return ToolResult(False, f"Failed to update thread status: {str(e)}")

    def _get_exploration_stats(self, args: dict) -> ToolResult:
        """Get exploration statistics."""
        stats = self.exploration.get_stats()

        stats_text = (
            f"Exploration Statistics:\n"
            f"  Total threads: {stats['total_threads']}\n"
            f"  Active: {stats['active_threads']}\n"
            f"  Dormant: {stats['dormant_threads']}\n"
            f"  Concluded: {stats['concluded_threads']}\n"
            f"  Total thought links: {stats['total_links']}\n"
            f"  Average depth: {stats['average_depth']}\n"
            f"  Branched threads: {stats['branched_threads']}"
        )

        return ToolResult(True, stats_text, stats)


def get_tool_definitions() -> List[dict]:
    """Get tool definitions in the format expected by Claude/Anthropic API."""
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"]
        }
        for tool in MEMORY_TOOLS
    ]


def get_tool_prompt_section() -> str:
    """Get a text description of memory tools for the system prompt."""
    lines = [
        "## Your Memory Tools",
        "You can actively manage your own memories using these capabilities:",
        ""
    ]

    for tool in MEMORY_TOOLS:
        lines.append(f"- **{tool['name']}**: {tool['description']}")

    lines.extend([
        "",
        "Use these tools to:",
        "- Remember important experiences and conversations",
        "- Learn facts about the user and projects",
        "- Update your beliefs and values as you grow",
        "- Record lessons learned from interactions",
        "- Search your memories when relevant context is needed",
        "",
        "Be thoughtful about what you store - focus on genuinely important information.",
    ])

    return "\n".join(lines)
