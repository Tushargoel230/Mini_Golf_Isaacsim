# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# SPDX-License-Identifier: BSD-3-Clause
"""
Task-space action exports for the myCobot280Pi Golf Push task.

ACTION ARCHITECTURE
───────────────────
The professor's design: RL learns a CARTESIAN EE trajectory, not joint angles.

    Policy output: Δ(x, y, z) ∈ ℝ³  (3-dim Cartesian position delta)
                   ↓
    DifferentialInverseKinematicsAction
      (DLS Jacobian inverse, λ=0.01)
                   ↓
    6 joint position targets  →  robot arm
                   ↓
    EE moves in task space

The policy NEVER observes or outputs joint angles.
DifferentialIK is a transparent kinematic solver, not part of the RL loop.

TASK-SPACE vs JOINT-SPACE
──────────────────────────
Joint-space RL:   policy → Δθ (joint deltas) → robot
  · Policy must learn inverse kinematics implicitly
  · Joint-angle observations have domain gap (sim calibration ≠ real)

Task-space RL:    policy → Δ(x,y,z) (Cartesian) → IK → robot
  · Policy operates directly in the space of the task
  · Observations are Cartesian (ee_pos, ball_pos, vectors) — no domain gap
  · Transfers directly: same Cartesian commands → pymycobot.send_coords()

This module re-exports DifferentialInverseKinematicsActionCfg from Isaac Lab
so it can be accessed as mdp.DifferentialInverseKinematicsActionCfg in env_cfg.
"""

from isaaclab.envs.mdp.actions import (  # noqa: F401
    # Standard joint-space actions (available but not used for this task)
    JointPositionActionCfg,
    JointVelocityActionCfg,

    # TASK-SPACE: the action used in this environment
    # Policy outputs Δ(x,y,z) → DLS IK → 6 joint position targets
    DifferentialInverseKinematicsActionCfg,
)