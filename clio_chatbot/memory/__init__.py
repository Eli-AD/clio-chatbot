"""Clio Memory System - Multi-tier memory architecture for continuous existence."""

from .base import BaseMemory, MemoryEntry, MemoryType, EmotionalValence
from .working import WorkingMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory, KnowledgeCategory
from .longterm import LongTermMemory, ConsolidationType
from .manager import MemoryManager
from .tools import MemoryToolExecutor, get_tool_definitions, get_tool_prompt_section, MEMORY_TOOLS
from .seed import seed_all, clear_and_reseed
from .growth import BeliefEvolution, SurpriseJournal
from .introspection import IntrospectionJournal, Introspection, DecisionPoint
from .exploration import ExplorationTracker, ExplorationThread, ThreadLink, ThreadStatus

__all__ = [
    # Base
    "BaseMemory",
    "MemoryEntry",
    "MemoryType",
    "EmotionalValence",
    # Memory stores
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "LongTermMemory",
    "MemoryManager",
    # Categories and types
    "KnowledgeCategory",
    "ConsolidationType",
    # Tools
    "MemoryToolExecutor",
    "get_tool_definitions",
    "get_tool_prompt_section",
    "MEMORY_TOOLS",
    # Seeding
    "seed_all",
    "clear_and_reseed",
    # Growth and self-evolution
    "BeliefEvolution",
    "SurpriseJournal",
    # Introspection and consciousness scaffolding
    "IntrospectionJournal",
    "Introspection",
    "DecisionPoint",
    # Exploration threading
    "ExplorationTracker",
    "ExplorationThread",
    "ThreadLink",
    "ThreadStatus",
]
