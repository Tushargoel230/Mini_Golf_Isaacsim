# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# SPDX-License-Identifier: BSD-3-Clause
"""
Action term exports for the myCobot280Pi Golf Push task.

The task uses DifferentialInverseKinematicsActionCfg exclusively:
  · Policy outputs 3D Cartesian deltas (Δx, Δy, Δz) in EE body frame
  · IK solver (DLS, λ=0.01) maps these to 6 joint position targets
  · Arm tracks targets via HIGH_PD actuators (stiffness=800, damping=50)

Scale 0.04 m/step at 25 Hz → max EE speed 1.0 m/s (safe for real robot).
body_offset (0,0,-0.012) tracks the physical contact tip of joint6_flange
(the URDF mesh origin is 12mm below the link frame origin).
"""

from isaaclab.envs.mdp.actions import (  # noqa: F401
    JointPositionActionCfg,
    JointVelocityActionCfg,
    DifferentialInverseKinematicsActionCfg,
)