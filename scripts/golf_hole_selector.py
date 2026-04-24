#!/usr/bin/env python3
# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# SPDX-License-Identifier: BSD-3-Clause
"""
Golf hole selector — run in a plain terminal (no Isaac Lab, no ROS2 needed).

Type 1, 2, or 3 to select which hole the robot aims for.
The change is picked up by golf_play_holes.py at the next episode reset.

Usage:
    python scripts/golf_hole_selector.py
"""

import json
import os

HOLE_FILE = "/tmp/golf_hole.json"

HOLE_LABELS = {
    1: "right  (85°)",
    2: "center (90°)",
    3: "left   (95°)",
}


def write_hole(hole_id: int) -> None:
    """Atomically write hole selection so Isaac never reads a partial file."""
    tmp = HOLE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"hole_id": hole_id}, f)
    os.replace(tmp, HOLE_FILE)   # atomic on Linux


def main() -> None:
    # Write a safe default so golf_play_holes.py finds the file on startup
    write_hole(2)
    print("Golf Hole Selector")
    print("──────────────────")
    print("  1 → right  hole  (85°)")
    print("  2 → center hole  (90°)  ← default")
    print("  3 → left   hole  (95°)")
    print()
    print("Changes take effect at the NEXT episode reset in Isaac Sim.")
    print("Press Ctrl-C or type 'q' to quit.\n")

    current = 2
    while True:
        try:
            s = input(f"Hole [{current}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if s in ("q", "quit", "exit"):
            print("Bye.")
            break
        if s in ("1", "2", "3"):
            current = int(s)
            write_hole(current)
            print(f"  ✓ Set to hole {current}: {HOLE_LABELS[current]}")
        elif s == "":
            pass   # re-show prompt with same hole
        else:
            print("  Enter 1, 2, or 3  (or q to quit)")


if __name__ == "__main__":
    main()
