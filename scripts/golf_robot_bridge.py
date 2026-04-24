#!/usr/bin/env python3
# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# SPDX-License-Identifier: BSD-3-Clause
"""
Golf robot bridge — run ON the myCobot280 Pi.

Subscribes to /mycobot280_tushar/joint_commands (Float64MultiArray: 6 joint
angles in degrees) and forwards them to the physical arm via pymycobot.
Also publishes the robot's actual joint angles back on /mycobot280_tushar/joint_states
at 10 Hz for monitoring.

Usage (on the Pi):
    source /opt/ros/humble/setup.bash
    python3 golf_robot_bridge.py [--speed 30] [--port /dev/serial0]
"""

import argparse

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

TOPIC_CMD   = "/mycobot280_tushar/joint_commands"
TOPIC_STATE = "/mycobot280_tushar/joint_states"

# myCobot280 hardware joint limits (degrees) — exact pymycobot firmware limits.
# Derived from URDF joint ranges and confirmed against live pymycobot errors.
# All values are tighter than ±170 — the bridge MUST clamp here before send_angles().
JOINT_LIMITS = [
    (-168.0, 168.0),  # J1 — robot_info.py angles_min/max[0]
    (-140.0, 140.0),  # J2 — robot_info.py angles_min/max[1]  (NOT ±135)
    (-150.0, 150.0),  # J3 — robot_info.py angles_min/max[2]
    (-150.0, 150.0),  # J4 — robot_info.py angles_min/max[3]  (NOT ±145)
    (-155.0, 160.0),  # J5 — robot_info.py asymmetric -155~160
    (-180.0, 180.0),  # J6 — robot_info.py angles_min/max[5]
]


class GolfRobotBridge(Node):

    def __init__(self, port: str, baud: int, speed: int):
        super().__init__("golf_robot_bridge")
        self.speed = speed

        # Connect to the arm — matches the working dance.py pattern
        from pymycobot.mycobot280 import MyCobot280
        self.mc = MyCobot280(port, baud)
        self.mc.set_fresh_mode(1)   # always execute latest command, discard queued ones
        self.get_logger().info(f"Connected to myCobot280 on {port} @ {baud}")

        # Receive joint commands from Isaac Sim
        self.sub = self.create_subscription(
            Float64MultiArray, TOPIC_CMD, self._on_cmd, 10
        )

        # Publish actual joint feedback at 20 Hz
        self.pub   = self.create_publisher(Float64MultiArray, TOPIC_STATE, 10)
        self.timer = self.create_timer(0.05, self._publish_state)

        self.get_logger().info(f"Listening  → {TOPIC_CMD}")
        self.get_logger().info(f"Publishing → {TOPIC_STATE}")

    def _on_cmd(self, msg: Float64MultiArray) -> None:
        if len(msg.data) not in (6, 7):
            self.get_logger().warn(
                f"Expected 6 or 7 values, got {len(msg.data)} — ignored."
            )
            return

        # Clamp to hardware limits before sending
        angles = [
            max(lo, min(hi, float(a)))
            for a, (lo, hi) in zip(msg.data[:6], JOINT_LIMITS)
        ]

        # Use per-command speed if relay encoded it as 7th element
        speed = int(msg.data[6]) if len(msg.data) == 7 else self.speed
        speed = max(1, min(100, speed))   # clamp to valid range

        try:
            self.mc.send_angles(angles, speed)
        except Exception as exc:
            self.get_logger().error(f"send_angles failed: {exc}")

    def _publish_state(self) -> None:
        try:
            angles = self.mc.get_angles()
            if angles:
                msg      = Float64MultiArray()
                msg.data = [float(a) for a in angles]
                self.pub.publish(msg)
        except Exception:
            pass   # robot may not respond during fast motion — skip silently


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  default="/dev/serial0",
                        help="Serial port (default: /dev/serial0)")
    parser.add_argument("--baud",  type=int, default=1000000)
    parser.add_argument("--speed", type=int, default=30,
                        help="Motor speed 0-100 (30 ≈ smooth demo speed).")
    args = parser.parse_args()

    rclpy.init()
    node = GolfRobotBridge(args.port, args.baud, args.speed)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down bridge.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
