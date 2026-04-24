# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# SPDX-License-Identifier: BSD-3-Clause
"""
MDP package — myCobot280Pi Golf Push (task-space RL, swing-based).

Exports every function/class used as mdp.X in mini_g_env_cfg.py.

Observation space: 18-dim (pure task space, no joint angles)
  ee_pos(3) + ball_pos(3) + ball_vel(3) + ee_to_ball(3)
  + ball_to_hole(3) + ee_to_backswing(3) = 18

Action space: 3-dim Cartesian Δ(x,y,z) via DifferentialIK
"""

# ── Observations (task-space, no joint angles) ────────────────────────────────
from .observations import (  # noqa: F401
    body_pos_w,
    object_position_in_robot_root_frame,
    object_lin_vel_w,
    ee_to_object_vector,
    ee_to_backswing,           # USED: guides EE to backswing position behind ball
    ball_to_hole_vector,       # USED: dynamic world-frame hole direction
    joint_pos_rel,             # kept for optional use
    joint_vel_rel,             # kept for optional use
)

# ── Rewards ───────────────────────────────────────────────────────────────────
from .rewards import (  # noqa: F401
    ee_approach_ball,          # kept — available at low weight if needed
    ee_height_match,           # w=1.5 — EE must swing at ball height
    ball_velocity_toward_hole, # w=12.0 PRIMARY push signal
    ball_to_hole_reward,       # w=6.0  displacement progress
    hole_in_one_bonus,         # w=200  terminal
    first_contact_approach,    # w=3.0  approach ball only pre-hit
    second_contact_penalty,    # w=-8.0 penalize re-hit after ball moving
    action_rate_l2,            # w=+0.01 penalizes action changes → one decisive swing
    joint_pos_limits,          # penalty: safety constraint
)

# ── Terminations ──────────────────────────────────────────────────────────────
from .terminations import (  # noqa: F401
    ball_in_hole,
    ball_out_of_bounds,
)
from isaaclab.envs.mdp.terminations import time_out  # noqa: F401

# ── Events ────────────────────────────────────────────────────────────────────
from .events import (  # noqa: F401
    disable_arm_collision_except_ee,  # startup: disable all arm links except EE
    disable_base_link_collision,      # alias for above (backward compat)
    open_camera_viewport,             # startup: open top-down camera panel
    reset_hole_position,              # reset: randomise hole in forward arc
    reset_arm_for_swing,              # reset: set arm behind ball, j6 toward hole
    reset_arm_to_prepush,             # kept for backward compat
    reset_ball_in_annulus,            # kept for optional use
    reset_joints_by_offset,           # kept for optional use
    reset_scene_to_default,           # reset: restore assets from init_state
)

# ── Actions (task-space) ──────────────────────────────────────────────────────
from .actions import (  # noqa: F401
    JointPositionActionCfg,
    JointVelocityActionCfg,
    DifferentialInverseKinematicsActionCfg,  # USED: policy → Δ(x,y,z) → IK → joints
)