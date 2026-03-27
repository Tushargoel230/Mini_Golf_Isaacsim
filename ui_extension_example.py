# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# SPDX-License-Identifier: BSD-3-Clause
"""
MDP package for the myCobot280Pi Golf Push task.

Exports all observations, rewards, terminations, events and actions
from a single namespace so env_cfg.py can write `mdp.function_name`.

Observation space (30-dim):
  joint_pos          6
  joint_vel          6  (× 0.1 scale applied in ObsTerm)
  ee_pos             3
  ball_pos           3  (from camera / physics state)
  ball_vel           3  (× 0.5 scale applied in ObsTerm)
  ee_to_ball         3
  ball_to_hole       3
  ee_to_approach     3  (IK approach target vector)
                    ──
  total             30

Action space (3-dim):
  Δx, Δy, Δz   EE position deltas via DifferentialIK  (scale 0.04 m/step)
"""

# ── Observations ──────────────────────────────────────────────────────────────
from .observations import (  # noqa: F401
    body_pos_w,
    object_position_in_robot_root_frame,
    object_lin_vel_w,
    ee_to_object_vector,
    ee_to_approach_target,
    ball_to_hole_vector,
    joint_pos_rel,
    joint_vel_rel,
)

# ── Rewards ───────────────────────────────────────────────────────────────────
from .rewards import (  # noqa: F401
    ee_approach_ball,
    ee_height_match,
    ball_velocity_toward_hole,
    ball_to_hole_reward,
    hole_in_one_bonus,
    action_rate_l2,
    joint_pos_limits,
)

# ── Terminations ──────────────────────────────────────────────────────────────
from .terminations import (  # noqa: F401
    ball_in_hole,
    ball_out_of_bounds,
)
from isaaclab.envs.mdp.terminations import time_out  # noqa: F401

# ── Events / Resets ───────────────────────────────────────────────────────────
from .events import (  # noqa: F401
    disable_base_link_collision,
    open_camera_viewport,
    reset_ball_in_annulus,
    reset_joints_by_offset,
    reset_scene_to_default,
)

# ── Actions ───────────────────────────────────────────────────────────────────
from .actions import (  # noqa: F401
    JointPositionActionCfg,
    JointVelocityActionCfg,
    DifferentialInverseKinematicsActionCfg,
)