"""Memory Tools - Functions the LLM can call to manage its own memories."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .base import MemoryType, EmotionalValence
from .semantic import KnowledgeCategory
from .longterm import ConsolidationType
from .manager import MemoryManager
from .growth import BeliefEvolution, SurpriseJournal


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
    }
]


class MemoryToolExecutor:
    """Executes memory tools called by the LLM."""

    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager

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
