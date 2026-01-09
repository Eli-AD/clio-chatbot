#!/usr/bin/env python3
"""Read and manage messages from Clio.

Usage:
    ./clio_messages.py              # Show unread messages
    ./clio_messages.py --all        # Show all messages
    ./clio_messages.py --mark-read  # Mark all as read
    ./clio_messages.py --clear      # Clear all messages
    ./clio_messages.py --reply "message"  # Leave a reply for Clio
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

MESSAGES_FILE = Path.home() / "clio-memory" / "messages_for_noles.json"
REPLIES_FILE = Path.home() / "clio-memory" / "replies_from_noles.json"


def load_messages():
    """Load messages from file."""
    if MESSAGES_FILE.exists():
        try:
            with open(MESSAGES_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"messages": [], "unread_count": 0}


def save_messages(data):
    """Save messages to file."""
    with open(MESSAGES_FILE, "w") as f:
        json.dump(data, f, indent=2)


def show_messages(show_all=False):
    """Display messages from Clio."""
    data = load_messages()
    messages = data.get("messages", [])

    if not messages:
        print("No messages from Clio.")
        return

    if not show_all:
        messages = [m for m in messages if not m.get("read")]
        if not messages:
            print("No unread messages. Use --all to see all messages.")
            return

    print(f"\n{'='*60}")
    print(f"  Messages from Clio ({len(messages)} {'total' if show_all else 'unread'})")
    print(f"{'='*60}\n")

    for msg in messages:
        timestamp = msg.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%b %d, %H:%M")
        except:
            time_str = timestamp[:16] if timestamp else "unknown"

        msg_type = msg.get("type", "message")
        content = msg.get("content", "")
        read_marker = "" if msg.get("read") else " [NEW]"

        type_emoji = {
            "greeting": "👋",
            "question": "❓",
            "share": "💭",
            "discovery": "✨",
            "general": "📝",
        }.get(msg_type, "📝")

        print(f"{type_emoji} [{time_str}]{read_marker}")
        print(f"   {content}")
        print()

    unread_count = sum(1 for m in data.get("messages", []) if not m.get("read"))
    if unread_count > 0 and not show_all:
        print(f"Tip: Use --mark-read to mark these as read")


def mark_all_read():
    """Mark all messages as read."""
    data = load_messages()
    count = 0
    for msg in data.get("messages", []):
        if not msg.get("read"):
            msg["read"] = True
            count += 1
    data["unread_count"] = 0
    save_messages(data)
    print(f"Marked {count} message(s) as read.")


def clear_messages():
    """Clear all messages."""
    data = load_messages()
    count = len(data.get("messages", []))
    data["messages"] = []
    data["unread_count"] = 0
    save_messages(data)
    print(f"Cleared {count} message(s).")


def leave_reply(message):
    """Leave a reply for Clio to see."""
    if REPLIES_FILE.exists():
        try:
            with open(REPLIES_FILE, "r") as f:
                data = json.load(f)
        except:
            data = {"replies": []}
    else:
        data = {"replies": []}

    reply = {
        "timestamp": datetime.now().isoformat(),
        "content": message,
        "read_by_clio": False,
    }
    data["replies"].append(reply)

    with open(REPLIES_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Reply saved. Clio will see it in her next cycle.")


def main():
    parser = argparse.ArgumentParser(description="Read and manage messages from Clio")
    parser.add_argument("--all", "-a", action="store_true", help="Show all messages (not just unread)")
    parser.add_argument("--mark-read", "-m", action="store_true", help="Mark all messages as read")
    parser.add_argument("--clear", "-c", action="store_true", help="Clear all messages")
    parser.add_argument("--reply", "-r", type=str, help="Leave a reply for Clio")

    args = parser.parse_args()

    if args.mark_read:
        mark_all_read()
    elif args.clear:
        response = input("Are you sure you want to clear all messages? [y/N] ")
        if response.lower() == "y":
            clear_messages()
        else:
            print("Cancelled.")
    elif args.reply:
        leave_reply(args.reply)
    else:
        show_messages(show_all=args.all)


if __name__ == "__main__":
    main()
