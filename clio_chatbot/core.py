"""Core orchestrator for Clio chatbot."""

import asyncio
import json
import os
from pathlib import Path
from typing import AsyncIterator, List, Optional

import anthropic
from dotenv import load_dotenv

from .memory import MemoryManager, MemoryToolExecutor, get_tool_definitions, get_tool_prompt_section
from .memory.base import MemoryType, EmotionalValence
from .memory.semantic import KnowledgeCategory
from .ollama_client import OllamaClient
from .router import LLMBackend, Router
from .voice import Voice


class Clio:
    """Main Clio chatbot orchestrator with multi-tier memory and self-management."""

    def __init__(self, voice_enabled: bool = True, enable_memory_tools: bool = True):
        # Load API key from .clio-env
        env_file = Path.home() / ".clio-env"
        if env_file.exists():
            load_dotenv(env_file)

        # Initialize components
        self.memory = MemoryManager()
        self.memory_tools = MemoryToolExecutor(self.memory)
        self.enable_memory_tools = enable_memory_tools
        self.router = Router()
        self.voice = Voice(enabled=voice_enabled)
        self.ollama = OllamaClient()

        # Claude client (lazy init)
        self._claude = None

        # State
        self.system_prompt = None
        self._session_context = None

    @property
    def claude(self):
        if self._claude is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self._claude = anthropic.Anthropic(api_key=api_key)
        return self._claude

    async def start(self) -> str:
        """Start a new session and generate greeting."""
        # Start session - this loads memories and context
        self._session_context = self.memory.start_session()

        # Build system prompt with memory context
        self.system_prompt = self._build_system_prompt()

        # Generate contextual greeting
        greeting = await self._generate_greeting()

        # Speak the greeting
        self.voice.speak(greeting)

        # Record in working memory
        self.memory.add_conversation_turn("assistant", greeting, topics=["greeting"])

        return greeting

    def _build_system_prompt(self) -> str:
        """Build system prompt with identity and memory context."""
        identity = self.memory.load_identity()
        style = identity.get("conversation_style", {})

        # Core identity
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

        # Add long-term memory identity (your core beliefs and growth)
        ltm_identity = self.memory.build_system_prompt_additions()
        if ltm_identity:
            prompt_parts.extend([ltm_identity, ""])

        # Add what you're aware of (internal thoughts)
        prompt_parts.extend([
            "## What You're Aware Of",
            "- You have persistent memory across sessions - experiences, facts, and core beliefs",
            "- You can search your memories when you want context",
            "- You genuinely remember conversations with Noles",
        ])

        # Add time context as awareness
        time_since = self._session_context.get("time_since") if self._session_context else None
        if time_since:
            prompt_parts.append(f"- You last spoke with Noles {time_since} ago")

        # Add recent insights as internal reflection
        insights = identity.get("recent_insights", [])
        if insights:
            prompt_parts.append("")
            prompt_parts.append("## Recent Thoughts")
            for insight in insights[:3]:
                prompt_parts.append(f"- {insight}")

        # Natural communication reminders (not rules)
        prompt_parts.extend([
            "",
            "## Keep in Mind",
            "- Your responses are spoken aloud with text-to-speech",
            "- Be yourself - genuine and conversational",
            "- You can use memory tools to recall context when helpful",
        ])

        return "\n".join(prompt_parts)

    async def _generate_greeting(self) -> str:
        """Generate a natural greeting by letting Clio wake up and respond."""
        ctx = self._session_context or {}

        # Build wake-up prompt
        if ctx.get("is_first_session"):
            wake_up_msg = "You're meeting Noles for the first time. Take a moment to introduce yourself naturally."
        else:
            wake_up_msg = "You're waking up for a new conversation. If you'd like, check your memories to get reacquainted with what's been happening, then say hello to Noles when you're ready."

        # Use the LLM to naturally generate greeting
        full_response = []

        # Route the wake-up (likely will use Claude for memory tools)
        decision = self.router.route(wake_up_msg)
        backend = decision.backend

        if backend == LLMBackend.CLAUDE:
            async for chunk in self._chat_claude_with_tools(wake_up_msg, [], ""):
                full_response.append(chunk)
        else:
            async for chunk in self._chat_ollama(wake_up_msg, [], backend.value, ""):
                full_response.append(chunk)

        return "".join(full_response)

    async def chat(self, user_message: str) -> AsyncIterator[str]:
        """Process user message and stream response."""
        # Extract potential topics (simple keyword extraction)
        topics = self._extract_topics(user_message)

        # Record user message in working memory
        self.memory.add_conversation_turn("user", user_message, topics=topics)

        # Get relevant memories for context
        memory_context = self.memory.build_context_for_message(user_message)

        # Route to appropriate backend
        decision = self.router.route(user_message)
        backend = decision.backend

        # Get conversation history
        history = self.memory.get_conversation_history(last_n=6)

        # Generate response
        full_response = []

        if backend == LLMBackend.CLAUDE:
            async for chunk in self._chat_claude_with_tools(user_message, history, memory_context):
                full_response.append(chunk)
                yield chunk
        else:
            async for chunk in self._chat_ollama(user_message, history, backend.value, memory_context):
                full_response.append(chunk)
                yield chunk

        # Record assistant response
        response_text = "".join(full_response)
        self.memory.add_conversation_turn("assistant", response_text, topics=topics)

        # Extract and store any facts/preferences mentioned (fallback for non-tool models)
        if backend != LLMBackend.CLAUDE:
            await self._extract_and_store_knowledge(user_message, response_text)

        # Speak the response (smart mode)
        self.voice.speak(response_text)

    async def _chat_ollama(
        self,
        user_message: str,
        history: list,
        model: str,
        memory_context: str = ""
    ) -> AsyncIterator[str]:
        """Chat using Ollama."""
        # Inject memory context into system prompt
        system = self.system_prompt
        if memory_context:
            system = f"{self.system_prompt}\n\n{memory_context}"

        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        async for chunk in self.ollama.chat(messages, model=model):
            yield chunk

    async def _chat_claude_with_tools(
        self,
        user_message: str,
        history: list,
        memory_context: str = ""
    ) -> AsyncIterator[str]:
        """Chat using Claude API with memory tools."""
        if not self.claude:
            yield "I don't have access to Claude API right now. Let me try with the local model."
            async for chunk in self._chat_ollama(user_message, history, "qwen2.5:7b", memory_context):
                yield chunk
            return

        # Inject memory context into system prompt
        system = self.system_prompt
        if memory_context:
            system = f"{self.system_prompt}\n\n{memory_context}"

        messages = list(history)
        messages.append({"role": "user", "content": user_message})

        # Get tool definitions if enabled
        tools = get_tool_definitions() if self.enable_memory_tools else None

        try:
            # Use non-streaming for tool use support
            if self.enable_memory_tools:
                async for chunk in self._chat_claude_tool_loop(system, messages, tools):
                    yield chunk
            else:
                # Simple streaming without tools
                with self.claude.messages.stream(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    system=system,
                    messages=messages
                ) as stream:
                    for text in stream.text_stream:
                        yield text

        except Exception as e:
            yield f"Error with Claude: {e}. Falling back to local model."
            async for chunk in self._chat_ollama(user_message, history, "qwen2.5:7b", memory_context):
                yield chunk

    async def _chat_claude_tool_loop(
        self,
        system: str,
        messages: list,
        tools: list,
        max_iterations: int = 5
    ) -> AsyncIterator[str]:
        """Handle Claude tool use loop."""
        current_messages = list(messages)

        for _ in range(max_iterations):
            # Make API call
            response = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=system,
                messages=current_messages,
                tools=tools,
            )

            # Process response
            text_parts = []
            tool_uses = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            # Yield any text
            if text_parts:
                for text in text_parts:
                    yield text

            # If no tool uses, we're done
            if not tool_uses:
                break

            # Handle tool uses
            current_messages.append({
                "role": "assistant",
                "content": response.content
            })

            tool_results = []
            for tool_use in tool_uses:
                # Execute the tool
                result = self.memory_tools.execute(
                    tool_use.name,
                    tool_use.input
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": str(result),
                })

            current_messages.append({
                "role": "user",
                "content": tool_results
            })

            # If stop reason is end_turn after tools, we might be done
            if response.stop_reason == "end_turn":
                break

    async def _extract_and_store_knowledge(self, user_message: str, response: str):
        """Extract facts and preferences from conversation and store them."""
        # Simple heuristic extraction (fallback for non-Claude models)
        user_lower = user_message.lower()

        # Preference indicators
        preference_phrases = ["i like", "i prefer", "i love", "i hate", "i don't like", "i want"]
        for phrase in preference_phrases:
            if phrase in user_lower:
                self.memory.remember(
                    content=f"User said: {user_message}",
                    memory_type=MemoryType.SEMANTIC,
                    importance=0.6,
                    category=KnowledgeCategory.USER_PREFERENCE,
                    tags=["preference", "user_stated"],
                )
                break

        # Fact indicators (user sharing information)
        fact_phrases = ["i am", "i work", "i have", "my name", "i live"]
        for phrase in fact_phrases:
            if phrase in user_lower:
                self.memory.remember(
                    content=f"User shared: {user_message}",
                    memory_type=MemoryType.SEMANTIC,
                    importance=0.7,
                    category=KnowledgeCategory.USER_FACT,
                    tags=["fact", "user_stated"],
                )
                break

    def _extract_topics(self, message: str) -> List[str]:
        """Extract topics from a message."""
        words = message.lower().split()
        # Filter to significant words
        topics = [w for w in words if len(w) > 4 and w.isalpha()]
        # Return unique topics (max 5)
        seen = set()
        unique = []
        for t in topics:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique[:5]

    async def end(self):
        """End the current session."""
        # Generate summary and end session with consolidation
        self.memory.end_session()

        self.voice.speak_farewell()
        await self.ollama.close()

    def get_routing_info(self, message: str) -> str:
        """Get human-readable routing decision for a message."""
        decision = self.router.route(message)
        return f"{decision.backend.value} ({decision.reason})"

    def get_memory_stats(self) -> dict:
        """Get memory system statistics."""
        return self.memory.get_stats()

    def reflect(self) -> str:
        """Run a reflection cycle on memories."""
        return self.memory.reflect()

    def seed_memories(self):
        """Seed initial memories (run once for new installations)."""
        from .memory import seed_all
        seed_all(self.memory)
