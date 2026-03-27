# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# SPDX-License-Identifier: BSD-3-Clause
"""
Observation functions for the myCobot280Pi Golf Push task.

All observations are in WORLD FRAME (not env-local) because:
  · The policy normalises inputs via RunningStandardScaler
  · World positions remain comparable across parallel envs after scaling
  · The ROS2 real-robot bridge provides world-frame ball positions from the
    top-down camera, so the interface is identical sim → real

Observation space (30-dim):
  joint_pos         (6)   current joint angles (rad)
  joint_vel         (6)   joint velocities × 0.1  (rad/s, scaled)
  ee_pos            (3)   EE world position (m)
  ball_pos          (3)   ball world position  ← camera / physics state
  ball_vel          (3)   ball linear velocity × 0.5  ← contact/push signal
  ee_to_ball        (3)   vector EE → ball (m)
  ball_to_hole      (3)   vector ball → hole (m)
  ee_to_approach    (3)   vector EE → IK approach target (ball + APPROACH_HEIGHT)
                    ────
  total            30
"""

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Passthrough from Isaac Lab (no wrapper needed)
from isaaclab.envs.mdp.observations import joint_pos_rel, joint_vel_rel  # noqa: F401


def body_pos_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """World-frame position of a specific body link on the robot.
    asset_cfg must have body_names set to the target link name.
    Returns (N, 3).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    idx = robot.find_bodies(asset_cfg.body_names)[0][0]
    return robot.data.body_pos_w[:, idx, :]   # (N, 3)


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Object position relative to robot root origin.
    Invariant to absolute env origin, consistent across parallel envs.
    Returns (N, 3).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj:   RigidObject  = env.scene[object_cfg.name]
    return obj.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]


def object_lin_vel_w(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """World-frame linear velocity of a rigid object.  Returns (N, 3)."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_lin_vel_w[:, :3]


def ee_to_object_vector(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """World-frame vector from EE to object.
    Gives the policy direct knowledge of how far EE is from the ball.
    Returns (N, 3).
    """
    robot: Articulation = env.scene[ee_cfg.name]
    obj:   RigidObject  = env.scene[object_cfg.name]
    idx = robot.find_bodies(ee_cfg.body_names)[0][0]
    ee_pos  = robot.data.body_pos_w[:, idx, :]
    obj_pos = obj.data.root_pos_w[:, :3]
    return obj_pos - ee_pos


def ball_to_hole_vector(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    hole_pos: tuple[float, float, float],
) -> torch.Tensor:
    """World-frame vector from ball to hole centre.  Returns (N, 3)."""
    obj: RigidObject = env.scene[object_cfg.name]
    ball_pos = obj.data.root_pos_w[:, :3]
    hole_t   = torch.tensor(hole_pos, device=env.device).unsqueeze(0)
    return hole_t - ball_pos


def ee_to_approach_target(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    approach_height: float = 0.08,
) -> torch.Tensor:
    """World-frame vector from EE to the IK approach target.

    Approach target = ball_world_pos + [0, 0, approach_height]

    This term encodes FK/IK knowledge directly in the observation:
    · When near zero → EE is directly above ball, ready to descend.
    · Policy learns: zero this first (reach phase) → zero ee_to_ball
      (descent+contact phase) → push toward hole (velocity reward).

    In sim:  ball_pos comes from physics state (perfect).
    In real: ball_pos comes from the top-down USB camera detection.
    The interface is identical, making sim-to-real transfer direct.

    Returns (N, 3).
    """
    robot: Articulation = env.scene[ee_cfg.name]
    obj:   RigidObject  = env.scene[object_cfg.name]
    idx     = robot.find_bodies(ee_cfg.body_names)[0][0]
    ee_pos  = robot.data.body_pos_w[:, idx, :]          # (N, 3)
    ball_pos = obj.data.root_pos_w[:, :3]               # (N, 3)
    approach = ball_pos.clone()
    approach[:, 2] = approach[:, 2] + approach_height   # above ball
    return approach - ee_pos                             # (N, 3)