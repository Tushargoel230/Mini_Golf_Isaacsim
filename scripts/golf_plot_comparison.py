#!/usr/bin/env python3
# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# SPDX-License-Identifier: BSD-3-Clause
"""
Real-time sim vs real joint comparison plot.

Reads commanded joints from /tmp/golf_joints.json (written by Isaac Sim) and
subscribes to /mycobot280_tushar/joint_states (published by the Pi bridge) to
show tracking quality for all 6 joints live.

Run with system Python 3.10 (NOT the Isaac Lab conda env):
    export ROS_DOMAIN_ID=44
    source /opt/ros/humble/setup.bash
    /usr/bin/python3 scripts/golf_plot_comparison.py [--save]

Options:
    --save   Auto-save all data to a timestamped CSV when the window closes.

Install matplotlib if needed:
    /usr/bin/pip3 install matplotlib
"""

import argparse
import csv
import json
import os
import threading
import time
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# ── CLI ───────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--save", action="store_true",
                     help="Save collected data to a CSV file on exit.")
_parser.add_argument("--outdir", default="logs/tracking",
                     help="Directory for CSV files (default: logs/tracking).")
_cli = _parser.parse_args()

# ── Constants ─────────────────────────────────────────────────────────────────
JOINTS_FILE  = "/tmp/golf_joints.json"
TOPIC_STATES = "/mycobot280_tushar/joint_states"
WINDOW_SEC   = 15
MAXLEN       = WINDOW_SEC * 30    # 15 s @ 30 Hz
JOINT_NAMES  = ["J1 (base)", "J2 (shoulder)", "J3 (elbow)",
                "J4 (wrist1)", "J5 (wrist2)", "J6 (ee)"]


# ── ROS2 Node ─────────────────────────────────────────────────────────────────

class GolfPlotter(Node):
    def __init__(self):
        super().__init__("golf_plotter")
        self.t0 = time.time()

        # Rolling buffers for the live plot (15 s window)
        self.t_sim  = deque(maxlen=MAXLEN)
        self.t_real = deque(maxlen=MAXLEN)
        self.sim_j  = [deque(maxlen=MAXLEN) for _ in range(6)]
        self.real_j = [deque(maxlen=MAXLEN) for _ in range(6)]

        # Full-history lists for CSV export (unbounded)
        self.csv_t_sim  = []
        self.csv_t_real = []
        self.csv_sim_j  = [[] for _ in range(6)]
        self.csv_real_j = [[] for _ in range(6)]

        # Subscribe to real robot joint states from Pi bridge
        self.create_subscription(
            Float64MultiArray, TOPIC_STATES, self._on_real, 10
        )

        # Poll sim joint file at 30 Hz
        self.create_timer(1.0 / 30, self._read_sim)

        self.get_logger().info(
            f"Plotting: sim ({JOINTS_FILE}) vs real ({TOPIC_STATES})"
        )

    def _on_real(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 6:
            t = time.time() - self.t0
            self.t_real.append(t)
            for i in range(6):
                self.real_j[i].append(float(msg.data[i]))
            if _cli.save:
                self.csv_t_real.append(t)
                for i in range(6):
                    self.csv_real_j[i].append(float(msg.data[i]))

    def _read_sim(self) -> None:
        try:
            with open(JOINTS_FILE) as f:
                angles = json.load(f).get("angles_deg", [])
            if len(angles) == 6:
                t = time.time() - self.t0
                self.t_sim.append(t)
                for i, a in enumerate(angles):
                    self.sim_j[i].append(float(a))
                if _cli.save:
                    self.csv_t_sim.append(t)
                    for i, a in enumerate(angles):
                        self.csv_sim_j[i].append(float(a))
        except Exception:
            pass   # file not yet written or mid-write


# ── matplotlib setup ──────────────────────────────────────────────────────────

def build_figure():
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle("Sim vs Real Joint Tracking", fontsize=13, fontweight="bold")

    lines_sim  = []
    lines_real = []

    for idx, ax in enumerate(axes.flat):
        ls, = ax.plot([], [], color="royalblue",  lw=1.8, label="sim")
        lr, = ax.plot([], [], color="darkorange", lw=1.5, ls="--", label="real")
        lines_sim.append(ls)
        lines_real.append(lr)

        ax.set_title(JOINT_NAMES[idx], fontsize=10)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("degrees", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes, lines_sim, lines_real


def make_update(node, fig, axes, lines_sim, lines_real):
    """Return the FuncAnimation update callback."""

    def _update(_frame):
        errors = []

        for i in range(6):
            t_s = list(node.t_sim)
            t_r = list(node.t_real)
            s   = list(node.sim_j[i])
            r   = list(node.real_j[i])

            lines_sim[i].set_data(t_s, s)
            lines_real[i].set_data(t_r, r)

            # Auto-scale axes
            ax = axes.flat[i]
            all_t = t_s + t_r
            all_v = s + r
            if all_t:
                ax.set_xlim(max(0, all_t[-1] - WINDOW_SEC), all_t[-1] + 0.5)
            if all_v:
                pad = max(5.0, (max(all_v) - min(all_v)) * 0.15)
                ax.set_ylim(min(all_v) - pad, max(all_v) + pad)

            # Per-joint error (last sample)
            if s and r:
                err = abs(s[-1] - r[-1])
                errors.append(err)
                ax.set_title(f"{JOINT_NAMES[i]}   err: {err:.1f}°", fontsize=9)
            else:
                ax.set_title(JOINT_NAMES[i], fontsize=9)

        # Overall RMS in suptitle
        if errors:
            rms = (sum(e**2 for e in errors) / len(errors)) ** 0.5
            fig.suptitle(
                f"Sim vs Real Joint Tracking   —   RMS error: {rms:.1f}°",
                fontsize=13, fontweight="bold"
            )

        return lines_sim + lines_real

    return _update


# ── CSV export ────────────────────────────────────────────────────────────────

def save_csv(node: GolfPlotter) -> None:
    """Write all collected sim + real joint data to a timestamped CSV file."""
    os.makedirs(_cli.outdir, exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(_cli.outdir, f"golf_tracking_{stamp}.csv")

    # Merge sim and real rows by source — each row tagged with source type
    rows = []
    for idx, t in enumerate(node.csv_t_sim):
        row = {"time_s": f"{t:.4f}", "source": "sim"}
        for j in range(6):
            row[f"j{j+1}_deg"] = f"{node.csv_sim_j[j][idx]:.4f}"
        rows.append(row)
    for idx, t in enumerate(node.csv_t_real):
        row = {"time_s": f"{t:.4f}", "source": "real"}
        for j in range(6):
            row[f"j{j+1}_deg"] = f"{node.csv_real_j[j][idx]:.4f}" if idx < len(node.csv_real_j[j]) else ""
        rows.append(row)

    rows.sort(key=lambda r: float(r["time_s"]))

    fieldnames = ["time_s", "source"] + [f"j{j+1}_deg" for j in range(6)]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    sim_pts  = len(node.csv_t_sim)
    real_pts = len(node.csv_t_real)
    print(f"[CSV] Saved {sim_pts} sim rows + {real_pts} real rows → {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    rclpy.init()
    node = GolfPlotter()

    # rclpy spins in background — matplotlib owns the main thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    fig, axes, lines_sim, lines_real = build_figure()
    update_fn = make_update(node, fig, axes, lines_sim, lines_real)

    ani = animation.FuncAnimation(     # noqa: F841  (kept alive by plt.show)
        fig, update_fn, interval=100, cache_frame_data=False
    )

    if _cli.save:
        print(f"[Plot] Data recording ON — CSV will be saved to {_cli.outdir}/ on exit.")
    print("[Plot] Window open — close it or press Ctrl-C to stop.")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        if _cli.save:
            save_csv(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
