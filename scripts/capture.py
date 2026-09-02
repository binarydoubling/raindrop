#!/usr/bin/env python3
"""Capture raindrop CLI output as SVG files for the README.

Usage:
    python scripts/capture.py                # capture all
    python scripts/capture.py current        # capture one command
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.text import Text

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

COMMANDS = [
    ["raindrop", "current", "Seattle"],
    ["raindrop", "hourly", "Seattle", "--hours", "12", "--spark"],
    ["raindrop", "daily", "Seattle"],
    ["raindrop", "route", "Seattle", "San Francisco", "-i", "100"],
    ["raindrop", "compare", "Seattle", "Portland", "San Francisco"],
    ["raindrop", "aqi", "Seattle"],
    ["raindrop", "astro", "Seattle"],
    ["raindrop", "marine", "San Diego"],
    ["raindrop", "clothing", "Seattle"],
]


def capture(args: list[str]) -> None:
    """Run a command and save its output as an SVG."""
    name = args[1]
    print(f"  Capturing {name}...", end=" ", flush=True)

    env = os.environ.copy()
    env["FORCE_COLOR"] = "1"
    env["TERM"] = "xterm-256color"

    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30, env=env)
        output = result.stdout or result.stderr
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return
    except OSError as e:
        print(f"ERROR: {e}")
        return

    if not output.strip():
        print("EMPTY")
        return

    console = Console(record=True, width=90, force_terminal=True)
    console.print(Text.from_ansi(output), end="")
    svg = console.export_svg(title=shlex.join(args))
    out_path = ASSETS / f"{name}.svg"
    out_path.write_text(svg)
    print(f"OK -> {out_path.relative_to(ROOT)}")


def main() -> None:
    """Capture requested commands, or every configured command."""
    ASSETS.mkdir(exist_ok=True)
    targets = sys.argv[1:]
    commands = [args for args in COMMANDS if not targets or args[1] in targets]
    if targets and len(commands) != len(set(targets)):
        available = {args[1] for args in COMMANDS}
        unknown = [target for target in targets if target not in available]
        print(f"Unknown command(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(sorted(available))}")
        raise SystemExit(1)

    print(f"Capturing {len(commands)} command(s):\n")
    for args in commands:
        capture(args)
    print("\nDone.")


if __name__ == "__main__":
    main()
