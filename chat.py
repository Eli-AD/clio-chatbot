#!/usr/bin/env python3
"""Clio Chat - CLI interface for the persistent AI companion."""

import asyncio
import sys

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from clio_chatbot.core import Clio

console = Console()


async def main():
    """Main chat loop."""
    console.print(Panel.fit(
        "[bold cyan]Clio[/bold cyan] - Your Persistent AI Companion",
        subtitle="Type 'quit' or 'exit' to end session"
    ))
    console.print()

    # Initialize Clio
    clio = Clio(voice_enabled=True)

    # Start session and show greeting
    console.print("[dim]Starting session...[/dim]")
    greeting = await clio.start()
    console.print(f"\n[bold cyan]Clio:[/bold cyan] {greeting}\n")

    # Main chat loop
    try:
        while True:
            # Get user input
            try:
                user_input = Prompt.ask("[bold green]You[/bold green]")
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input.strip():
                continue

            # Check for exit commands
            if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
                break

            # Check for special commands
            if user_input.startswith("/"):
                await handle_command(clio, user_input)
                continue

            # Get response with streaming
            console.print("[bold cyan]Clio:[/bold cyan] ", end="")

            response_parts = []
            async for chunk in clio.chat(user_input):
                console.print(chunk, end="")
                response_parts.append(chunk)

            console.print("\n")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

    finally:
        # End session gracefully
        console.print("\n[dim]Ending session...[/dim]")
        await clio.end()
        console.print("[dim]Session saved. See you next time![/dim]")


async def handle_command(clio: Clio, command: str):
    """Handle special commands."""
    cmd = command.lower().strip()

    if cmd == "/help":
        console.print("""
[bold]Commands:[/bold]
  /help     - Show this help
  /route    - Show how queries are routed
  /memory   - Show memory statistics
  /reflect  - Run memory reflection
  /seed     - Seed initial memories (run once)
  /voice    - Toggle voice on/off
  quit/exit - End session
""")

    elif cmd == "/voice":
        clio.voice.enabled = not clio.voice.enabled
        status = "enabled" if clio.voice.enabled else "disabled"
        console.print(f"[dim]Voice {status}[/dim]")

    elif cmd.startswith("/route "):
        query = command[7:]
        info = clio.get_routing_info(query)
        console.print(f"[dim]Would route to: {info}[/dim]")

    elif cmd == "/memory":
        stats = clio.get_memory_stats()
        console.print("[bold]Memory Statistics:[/bold]")
        console.print(f"  Episodic memories:  {stats.get('episodic_count', 0)}")
        console.print(f"  Semantic memories:  {stats.get('semantic_count', 0)}")
        console.print(f"  Long-term memories: {stats.get('longterm_count', 0)}")
        console.print(f"  Session turns:      {stats.get('working_conversation_turns', 0)}")
        console.print(f"  Retrieved memories: {stats.get('working_retrieved_memories', 0)}")

    elif cmd == "/reflect":
        console.print("[dim]Running reflection...[/dim]")
        result = clio.reflect()
        console.print(f"[dim]{result}[/dim]")

    elif cmd == "/seed":
        console.print("[dim]Seeding initial memories...[/dim]")
        clio.seed_memories()
        console.print("[dim]Initial memories seeded![/dim]")

    else:
        console.print(f"[dim]Unknown command: {command}[/dim]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted. Goodbye![/dim]")
        sys.exit(0)
