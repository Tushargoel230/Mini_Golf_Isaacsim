#!/usr/bin/env python3
# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# SPDX-License-Identifier: BSD-3-Clause
"""
golf_joint_relay.py

PURPOSE: Reads joint angles from /tmp/golf_joints.json and publishes to ROS2.
         Isaac Lab writes that file. No rclpy inside Isaac ever.

         This is the mirror of ros_goal_listener.py:
           ros_goal_listener: ROS2 topic → json file → Isaac Lab  (goals IN)
           golf_joint_relay:  Isaac Lab  → json file → ROS2 topic (joints OUT)

RUN THIS IN: A terminal with ROS2 sourced (NOT the Isaac terminal)

USAGE:
    source /opt/ros/humble/setup.bash
    /usr/bin/python3 scripts/golf_joint_relay.py
"""

import json
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

JOINTS_FILE = "/tmp/golf_joints.json"
TOPIC       = "/mycobot280_tushar/joint_commands"
RATE_HZ     = 10   # 10 Hz — 100 ms per command gives the real arm time to execute each step


class JointRelay(Node):
    def __init__(self):
        super().__init__("golf_joint_relay")

        self.pub = self.create_publisher(Float64MultiArray, TOPIC, 10)
        self.create_timer(1.0 / RATE_HZ, self._publish)

        self._last_angles = None

        # Clear stale file from any previous session to prevent flooding the
        # bridge with an outdated joint command the moment it connects.
        if os.path.exists(JOINTS_FILE):
            os.remove(JOINTS_FILE)
            self.get_logger().info(f"Cleared stale {JOINTS_FILE}")

        self.get_logger().info(f"Publishing → {TOPIC} at {RATE_HZ} Hz")
        self.get_logger().info(f"Reading from {JOINTS_FILE}")
        self.get_logger().info("Waiting for golf_play_ros2_bridge.py --ros2 ...")

    def _publish(self) -> None:
        try:
            with open(JOINTS_FILE) as f:
                data = json.load(f)
            angles = data.get("angles_deg", [])
            speed  = float(data.get("speed", 30))
            if len(angles) == 6 and angles != self._last_angles:
                msg      = Float64MultiArray()
                msg.data = [float(a) for a in angles] + [speed]  # 7 elements
                self.pub.publish(msg)
                self._last_angles = angles
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass   # file not yet written or mid-write — skip this tick


def main() -> None:
    rclpy.init()
    node = JointRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down relay.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
