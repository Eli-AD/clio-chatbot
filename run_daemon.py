#!/usr/bin/env python3
"""Run Clio's autonomous daemon.

This starts the daemon that gives Clio autonomous activity time
when you're not actively chatting with her.

Usage:
    ./run_daemon.py           # Run continuously
    ./run_daemon.py --once    # Run single cycle then exit
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from clio_chatbot.daemon import DaemonRunner


async def main():
    runner = DaemonRunner()

    if "--once" in sys.argv:
        print("Running single daemon cycle...")
        result = await runner.run_single_cycle()
        if result:
            print(f"Activity: {result.activity_type.value}")
            print(f"Success: {result.success}")
            print(f"Summary: {result.summary}")
        else:
            print("Cycle skipped (user active or outside hours)")
    else:
        print("Starting Clio daemon...")
        print(f"Cycle interval: {runner.config.cycle_interval}s")
        print("Press Ctrl+C to stop")
        print()
        await runner.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDaemon stopped.")
