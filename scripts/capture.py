#!/usr/bin/env python3
"""Capture raindrop CLI output as SVG files for the README.

Usage:
    python scripts/capture.py                # capture all
    python scripts/capture.py current        # capture one command
"""

import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.text import Text

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

# Commands to capture: (name, args, title)
COMMANDS = [
    ("current", ["raindrop", "current", "Seattle"], "raindrop current Seattle"),
    (
        "hourly",
        ["raindrop", "hourly", "Seattle", "--hours", "12", "--spark"],
        "raindrop hourly Seattle --hours 12 --spark",
    ),
    ("daily", ["raindrop", "daily", "Seattle"], "raindrop daily Seattle"),
    (
        "route",
        ["raindrop", "route", "Seattle", "San Francisco", "-i", "100"],
        "raindrop route Seattle 'San Francisco' -i 100",
    ),
    (
        "compare",
        ["raindrop", "compare", "Seattle", "Portland", "San Francisco"],
        "raindrop compare Seattle Portland 'San Francisco'",
    ),
    ("alerts", ["raindrop", "alerts", "Seattle"], "raindrop alerts Seattle"),
    ("aqi", ["raindrop", "aqi", "Seattle"], "raindrop aqi Seattle"),
    ("astro", ["raindrop", "astro", "Seattle"], "raindrop astro Seattle"),
    ("marine", ["raindrop", "marine", "San Diego"], "raindrop marine 'San Diego'"),
    ("clothing", ["raindrop", "clothing", "Seattle"], "raindrop clothing Seattle"),
    ("history", ["raindrop", "history", "Seattle"], "raindrop history Seattle"),
    (
        "precip",
        ["raindrop", "precip", "Seattle", "--days", "7"],
        "raindrop precip Seattle --days 7",
    ),
]


def capture(name: str, args: list[str], title: str) -> None:
    """Run a command and save its output as an SVG."""
    print(f"  Capturing {name}...", end=" ", flush=True)

    env = os.environ.copy()
    env["FORCE_COLOR"] = "1"
    env["TERM"] = "xterm-256color"

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        output = result.stdout
        if result.stderr and not output:
            output = result.stderr
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return
    except Exception as e:
        print(f"ERROR: {e}")
        return

    if not output.strip():
        print("EMPTY")
        return

    console = Console(record=True, width=90, force_terminal=True)
    text = Text.from_ansi(output)
    console.print(text, end="")

    svg = console.export_svg(title=title)
    out_path = ASSETS / f"{name}.svg"
    out_path.write_text(svg)
    print(f"OK -> {out_path.relative_to(ROOT)}")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)

    if len(sys.argv) > 1:
        targets = sys.argv[1:]
        commands = [c for c in COMMANDS if c[0] in targets]
        if not commands:
            print(f"Unknown command(s): {', '.join(targets)}")
            print(f"Available: {', '.join(c[0] for c in COMMANDS)}")
            sys.exit(1)
    else:
        commands = COMMANDS

    print(f"Capturing {len(commands)} command(s):\n")
    for name, args, title in commands:
        capture(name, args, title)
    print("\nDone.")


if __name__ == "__main__":
    main()
