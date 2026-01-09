#!/usr/bin/env python3
"""Clio Chat - CLI interface for the persistent AI companion."""

import argparse
import asyncio
import sys
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from clio_chatbot.core import Clio
from clio_chatbot.daemon import DaemonRunner

console = Console()

# Voice input (lazy loaded)
voice_input = None
voice_input_enabled = False  # Toggleable during session

# Daemon state
daemon_task = None
daemon_runner = None
daemon_message_queue = asyncio.Queue()


def init_voice_input():
    """Initialize voice input (lazy load to speed up startup)."""
    global voice_input
    if voice_input is None:
        console.print("[dim]Loading speech recognition...[/dim]")
        from clio_chatbot.voice_input import VoiceInput
        voice_input = VoiceInput()
        voice_input.load()
        console.print("[dim]Speech recognition ready![/dim]")
    return voice_input


def get_voice_input_text() -> str:
    """Get input via voice, returns the transcribed text."""
    vi = init_voice_input()

    # Simple status indicator - no fancy updating
    console.print("[dim]Listening... (speak now)[/dim]")

    text = vi.listen_once(timeout=120.0)

    if text:
        console.print(f"[bold green]You[/bold green]: {text}")
        return text
    else:
        console.print("[dim]No speech detected[/dim]")
        return ""


async def run_daemon_cycle(runner: DaemonRunner, clio: Clio):
    """Run the daemon in the background while chat is active."""
    # Override the reach_out handler to send to our queue instead of file
    original_send_message = runner.activity_handler.send_message_to_noles

    def send_to_chat(message: str, message_type: str = "general"):
        """Send message directly to chat instead of file."""
        asyncio.create_task(daemon_message_queue.put({
            "message": message,
            "type": "reach_out",
            "timestamp": datetime.now(),
        }))
        # Still log it
        from clio_chatbot.daemon.activities import ActivityResult, ActivityType
        result = ActivityResult(
            activity_type=ActivityType.REACH_OUT,
            success=True,
            summary=f"Sent message to chat: {message[:50]}...",
            details={"message": message, "type": message_type},
        )
        runner.activity_handler.log_activity(result)
        return result

    runner.activity_handler.send_message_to_noles = send_to_chat

    # Wait before first cycle to let user start chatting
    await asyncio.sleep(60)

    while True:
        try:
            # Only run if user has been idle
            if not runner._is_user_active():
                result = await runner.run_single_cycle()
                if result:
                    # Send activity status to chat
                    activity_labels = {
                        "introspect": "exploring internally",
                        "journal": "writing in journal",
                        "rest": "taking a moment to rest",
                        "reach_out": "reaching out",  # Already handled above
                        "web_search": "searching the web",
                    }
                    activity_name = result.activity_type.value

                    # Store activity in Clio's episodic memory
                    from clio_chatbot.memory.base import MemoryType, EmotionalValence

                    # Build memory content based on activity type
                    # NOTE: introspections are NOT stored as episodic (already in IntrospectionJournal)
                    if activity_name == "introspect":
                        # Don't create episodic memory - introspections are already richly stored
                        # in the introspection journal. Just use for conversation history.
                        # Record the ACTUAL THOUGHTS as Clio had them - not a report about thinking
                        thoughts = result.details.get('thoughts', '')
                        if thoughts:
                            # Direct thoughts, as experienced - not a meta-description
                            memory_content = f"*quietly reflecting* {thoughts}"
                        else:
                            question = result.details.get('question', '')
                            memory_content = f"*quietly reflecting* {question}"
                        store_episodic = False
                    elif activity_name == "journal":
                        memory_content = f"I wrote in my journal: {result.details.get('title', result.summary)}"
                        importance = 0.6
                        store_episodic = True
                    elif activity_name == "web_search":
                        query = result.details.get('query', '')
                        count = result.details.get('result_count', 0)
                        memory_content = f"I searched the web for '{query}' and found {count} results"
                        importance = 0.5
                        store_episodic = True
                    elif activity_name == "reach_out":
                        memory_content = f"I reached out to Noles: {result.details.get('message', '')[:60]}..."
                        importance = 0.8
                        store_episodic = True
                    elif activity_name == "rest":
                        reflection = result.details.get('reflection', '')
                        memory_content = f"I took a moment to rest{': ' + reflection if reflection else ''}"
                        importance = 0.4
                        store_episodic = True
                    else:
                        memory_content = result.summary
                        importance = 0.5
                        store_episodic = True

                    # Store as episodic memory (except introspections - already in introspection journal)
                    if store_episodic:
                        clio.memory.remember(
                            content=memory_content,
                            memory_type=MemoryType.EPISODIC,
                            importance=importance,
                            emotional_valence=EmotionalValence.NEUTRAL,
                            tags=["autonomous", activity_name],
                            source="daemon",
                        )

                    # ALSO add to conversation history so it's part of the 20-turn context
                    # This makes autonomous activities feel continuous with the conversation
                    # For introspections: record raw thoughts (no wrapper - feels like continuous inner life)
                    # For other activities: use [autonomous: ] wrapper to mark external actions
                    if activity_name == "introspect":
                        conversation_content = memory_content  # Direct thoughts, as experienced
                    else:
                        conversation_content = f"[autonomous: {memory_content}]"

                    clio.memory.add_conversation_turn(
                        role="assistant",
                        content=conversation_content,
                        topics=["autonomous", activity_name],
                    )

                    # CRITICAL: Save conversation immediately so Clio can recall autonomous activities
                    clio.memory._save_conversation()

                    # Only show status for non-reach-out activities (reach-out shows the actual message)
                    if activity_name != "reach_out":
                        status_message = activity_labels.get(activity_name, activity_name)
                        asyncio.create_task(daemon_message_queue.put({
                            "message": status_message,
                            "type": "activity_status",
                            "timestamp": datetime.now(),
                            "details": result.summary,
                        }))
        except asyncio.CancelledError:
            break
        except Exception as e:
            console.print(f"[dim]Daemon error: {e}[/dim]")

        # Wait for next cycle (5 minutes default)
        try:
            await asyncio.sleep(runner.config.cycle_interval)
        except asyncio.CancelledError:
            break


async def check_daemon_messages():
    """Check for and display any messages from daemon Clio."""
    try:
        while not daemon_message_queue.empty():
            msg = await daemon_message_queue.get()

            msg_type = msg.get("type", "activity_status")

            if msg_type == "reach_out":
                # Direct message from Clio
                console.print()
                console.print(f"[bold magenta]✨ Clio (autonomous):[/bold magenta] {msg['message']}")
                console.print()
            elif msg_type == "activity_status":
                # Activity status update
                console.print(f"[dim italic]*Clio is {msg['message']}*[/dim italic]")
    except Exception:
        pass


async def message_display_loop():
    """Background task that displays daemon messages in real-time."""
    while True:
        try:
            # Wait for a message with timeout
            msg = await asyncio.wait_for(daemon_message_queue.get(), timeout=1.0)

            msg_type = msg.get("type", "activity_status")

            if msg_type == "reach_out":
                # Direct message from Clio - make it noticeable
                console.print()
                console.print(f"\n[bold magenta]✨ Clio (autonomous):[/bold magenta] {msg['message']}")
                console.print()
            elif msg_type == "activity_status":
                # Activity status update
                console.print(f"\n[dim italic]*Clio is {msg['message']}*[/dim italic]")
        except asyncio.TimeoutError:
            # No message, just continue
            pass
        except asyncio.CancelledError:
            break
        except Exception:
            pass


def get_input_sync(prompt_text: str) -> str:
    """Synchronous input function to run in thread."""
    return Prompt.ask(prompt_text)


async def main(use_voice: bool = False, seamless: bool = False, enable_daemon: bool = True):
    """Main chat loop."""
    global voice_input_enabled, daemon_task, daemon_runner
    voice_input_enabled = use_voice

    subtitle = "Speak or Ctrl+C to exit" if use_voice else "Type 'quit' or 'exit' to end session"
    console.print(Panel.fit(
        "[bold cyan]Clio[/bold cyan] - Your Persistent AI Companion",
        subtitle=subtitle
    ))
    console.print()

    if use_voice:
        console.print("[dim]Voice mode enabled - speak to chat! Use /mic to toggle.[/dim]\n")

    # Initialize Clio
    clio = Clio(voice_enabled=True, seamless=seamless)

    # Start session
    if seamless:
        console.print("[dim]Continuing conversation...[/dim]\n")
        await clio.start()  # No greeting in seamless mode
    else:
        console.print("[dim]Starting session...[/dim]")
        greeting = await clio.start()
        if greeting:
            console.print(f"\n[bold cyan]Clio:[/bold cyan] {greeting}\n")

    # Start daemon in background if enabled
    message_task = None
    if enable_daemon:
        daemon_runner = DaemonRunner()
        if daemon_runner.client:
            daemon_task = asyncio.create_task(run_daemon_cycle(daemon_runner, clio))
            # Start background message display task
            message_task = asyncio.create_task(message_display_loop())
            console.print("[dim]Autonomous daemon active (cycles every 5 min when idle)[/dim]\n")
        else:
            console.print("[dim]Daemon disabled - no API key[/dim]\n")

    # Main chat loop
    try:
        while True:
            # Get user input (run in thread to allow async message display)
            try:
                if voice_input_enabled:
                    console.print("[dim](Say 'command mode' or wait for timeout to type instead)[/dim]")
                    user_input = get_voice_input_text()

                    # Check if user said "command mode" to switch to typing
                    if user_input and "command mode" in user_input.lower():
                        console.print("[dim]Switching to text input...[/dim]")
                        user_input = await asyncio.to_thread(get_input_sync, "[bold green]You[/bold green]")
                    # If no speech detected, offer text input
                    elif not user_input:
                        console.print("[dim]No speech detected - type your message:[/dim]")
                        user_input = await asyncio.to_thread(get_input_sync, "[bold green]You[/bold green]")
                else:
                    user_input = await asyncio.to_thread(get_input_sync, "[bold green]You[/bold green]")
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input or not user_input.strip():
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
        # Stop message display task if running
        if message_task:
            message_task.cancel()
            try:
                await message_task
            except asyncio.CancelledError:
                pass

        # Stop daemon if running
        if daemon_task:
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass
            console.print("[dim]Daemon stopped.[/dim]")

        # End session gracefully
        if clio.seamless:
            # Quiet exit in seamless mode - Clio doesn't perceive endings
            await clio.end()
        else:
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
  /voice    - Toggle voice output on/off
  /mic      - Toggle voice input on/off
  /daemon   - Show daemon status and recent activities
  /threads  - Show active exploration threads
  quit/exit - End session
""")

    elif cmd == "/voice":
        clio.voice.enabled = not clio.voice.enabled
        status = "enabled" if clio.voice.enabled else "disabled"
        console.print(f"[dim]Voice output {status}[/dim]")

    elif cmd == "/mic":
        global voice_input_enabled
        voice_input_enabled = not voice_input_enabled
        status = "enabled" if voice_input_enabled else "disabled"
        console.print(f"[dim]Voice input {status} (takes effect next prompt)[/dim]")
        if voice_input_enabled:
            init_voice_input()  # Pre-load the models
        else:
            console.print("[dim]Type your next message[/dim]")

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

    elif cmd == "/daemon":
        if daemon_runner:
            console.print("[bold]Daemon Status:[/bold]")
            console.print(f"  Running: {daemon_task is not None and not daemon_task.done()}")
            console.print(f"  Cycle interval: {daemon_runner.config.cycle_interval}s")

            # Show recent activities
            recent = daemon_runner.activity_handler.get_recent_activities(limit=5)
            if recent:
                console.print("\n[bold]Recent Autonomous Activities:[/bold]")
                for act in recent:
                    console.print(f"  - {act.get('activity_type')}: {act.get('summary', '')[:50]}...")
            else:
                console.print("  No autonomous activities yet")

            # Show exploration stats
            stats = daemon_runner.exploration_tracker.get_stats()
            console.print(f"\n[bold]Exploration Stats:[/bold]")
            console.print(f"  Active threads: {stats['active_threads']}")
            console.print(f"  Total thoughts: {stats['total_links']}")
        else:
            console.print("[dim]Daemon not initialized[/dim]")

    elif cmd == "/threads":
        if daemon_runner:
            threads = daemon_runner.exploration_tracker.list_active_threads(limit=10)
            if threads:
                console.print("[bold]Active Exploration Threads:[/bold]")
                for t in threads:
                    console.print(f"\n  [cyan]{t.name}[/cyan] (depth: {t.depth})")
                    console.print(f"    Question: {t.question[:60]}...")
            else:
                console.print("[dim]No active exploration threads[/dim]")
        else:
            console.print("[dim]Daemon not initialized[/dim]")

    else:
        console.print(f"[dim]Unknown command: {command}[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clio Chat - Your Persistent AI Companion")
    parser.add_argument("-t", "--type", action="store_true",
                        help="Use typing instead of voice input")
    parser.add_argument("--greet", action="store_true",
                        help="Enable greetings/farewells (off by default for seamless experience)")
    parser.add_argument("--no-daemon", action="store_true",
                        help="Disable autonomous daemon (runs by default)")
    args = parser.parse_args()

    try:
        asyncio.run(main(
            use_voice=not args.type,
            seamless=not args.greet,
            enable_daemon=not args.no_daemon
        ))
    except KeyboardInterrupt:
        console.print("\n[dim]Session saved.[/dim]")
        sys.exit(0)
