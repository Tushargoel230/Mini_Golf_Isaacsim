# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# SPDX-License-Identifier: BSD-3-Clause
"""
Reward functions for the myCobot280Pi Golf Push task.

Reward hierarchy  (four stages, all dense except hole_in_one):
───────────────────────────────────────────────────────────────
Stage 1a  ee_approach_target    EE → approach point above ball   weight  3.0
Stage 1b  ee_approach_ball      EE → ball (full 3D)              weight  2.0
Stage 1c  ee_height_match       EE z matches ball z              weight  1.5
Stage 2   ball_velocity_to_hole ball velocity toward hole        weight  8.0  ← key
Stage 3   ball_to_hole          ball distance to hole            weight  5.0
Stage 4   hole_in_one           sparse terminal bonus            weight 200.0
Penalty   action_rate_l2        smooth actions                   weight -0.01
Penalty   joint_pos_limits      stay away from joint limits      weight -1.0

Why ball_velocity_toward_hole is the key term
──────────────────────────────────────────────
reward = tanh(vel_scale · max(0, v_ball · ĥ_hole))

  v_ball : ball velocity vector
  ĥ_hole : unit vector from ball to hole (XY plane only)

This simultaneously teaches:
  · PUSH DIRECTION: dot product zero unless ball moves toward hole
  · PUSH FORCE:     higher speed = higher reward (saturated by tanh)
  · CONTACT:        fires only after the EE has actually hit the ball

The formula rewards any push toward the hole rather than any push along a
fixed direction, so the policy generalises to randomised ball positions.
"""

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── Stage 1a ──────────────────────────────────────────────────────────────────

def ee_approach_ball(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    k: float = 10.0,
) -> torch.Tensor:
    """Dense reward 1/(1+k·d): gradient everywhere across the workspace.

    Used for both the 'approach target above ball' and 'approach ball itself'
    terms (called twice with different k values from env_cfg RewardsCfg).

    At d=0.20m (typical start): 1/(1+10×0.20) = 0.33  (strong signal)
    At d=0.02m (near contact):  1/(1+10×0.02) = 0.83  (near peak)
    """
    robot: Articulation = env.scene[ee_cfg.name]
    ball:  RigidObject  = env.scene[object_cfg.name]
    idx     = robot.find_bodies(ee_cfg.body_names)[0][0]
    ee_pos  = robot.data.body_pos_w[:, idx, :]
    ball_pos = ball.data.root_pos_w[:, :3]
    dist = torch.linalg.norm(ball_pos - ee_pos, dim=-1)
    return 1.0 / (1.0 + k * dist)


# ── Stage 1c ──────────────────────────────────────────────────────────────────

def ee_height_match(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalise vertical gap between EE and ball.

    Critical for ground-level tasks: without this the arm stays high and
    never makes contact with the ball.  Returns ≤ 0.
    """
    robot: Articulation = env.scene[ee_cfg.name]
    ball:  RigidObject  = env.scene[object_cfg.name]
    idx   = robot.find_bodies(ee_cfg.body_names)[0][0]
    ee_z  = robot.data.body_pos_w[:, idx, 2]
    ball_z = ball.data.root_pos_w[:, 2]
    return -torch.abs(ee_z - ball_z)


# ── Stage 2 ───────────────────────────────────────────────────────────────────

def ball_velocity_toward_hole(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    hole_pos: tuple[float, float, float],
    vel_scale: float = 5.0,
) -> torch.Tensor:
    """Reward for ball velocity component directed toward the hole.

    Formula: tanh(vel_scale · max(0, v_ball · ĥ_hole))

    · XY only: z-velocity is irrelevant for a ball rolling on a flat surface
    · max(0,…): zero-rewards backward motion without penalising it
    · tanh saturates at ~0.2 m/s (with vel_scale=5) → no reward explosion
    · Fires only after contact: stationary ball gives zero reward

    Returns ∈ [0, 1].
    """
    ball: RigidObject = env.scene[object_cfg.name]
    ball_pos = ball.data.root_pos_w[:, :3]          # (N, 3)
    ball_vel = ball.data.root_lin_vel_w              # (N, 3)
    hole_t   = torch.tensor(hole_pos, device=env.device).unsqueeze(0)

    to_hole    = hole_t[:, :2] - ball_pos[:, :2]    # (N, 2)
    to_hole_n  = to_hole / (torch.linalg.norm(to_hole, dim=-1, keepdim=True) + 1e-6)
    vel_toward = (ball_vel[:, :2] * to_hole_n).sum(dim=-1)  # (N,)
    return torch.tanh(vel_scale * vel_toward.clamp(min=0.0))


# ── Stage 3 ───────────────────────────────────────────────────────────────────

def ball_to_hole_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    hole_pos: tuple[float, float, float],
    k: float = 6.0,
) -> torch.Tensor:
    """Dense reward 1/(1+k·d_xy) as ball approaches hole.

    XY distance only (z irrelevant for rolling ball).
    Returns ∈ (0, 1].
    """
    ball: RigidObject = env.scene[object_cfg.name]
    ball_pos = ball.data.root_pos_w[:, :3]
    hole_t   = torch.tensor(hole_pos, device=env.device).unsqueeze(0)
    dist_xy  = torch.linalg.norm(ball_pos[:, :2] - hole_t[:, :2], dim=-1)
    return 1.0 / (1.0 + k * dist_xy)


# ── Stage 4 ───────────────────────────────────────────────────────────────────

def hole_in_one_bonus(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    hole_pos: tuple[float, float, float],
    hole_radius: float = 0.042,
) -> torch.Tensor:
    """Sparse +1 when ball centre is within hole_radius of hole (XY plane).
    Returns ∈ {0, 1}.
    """
    ball: RigidObject = env.scene[object_cfg.name]
    ball_pos = ball.data.root_pos_w[:, :3]
    hole_t   = torch.tensor(hole_pos, device=env.device).unsqueeze(0)
    dist_xy  = torch.linalg.norm(ball_pos[:, :2] - hole_t[:, :2], dim=-1)
    return (dist_xy < hole_radius).float()


# ── Penalties ─────────────────────────────────────────────────────────────────

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 penalty on consecutive action change.
    Encourages smooth trajectories — critical for sim-to-real transfer.
    """
    return -torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action),
        dim=-1,
    )


def joint_pos_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    soft_ratio: float = 0.95,
) -> torch.Tensor:
    """Penalty proportional to how close joints are to their soft limits.
    Prevents the policy from exploring dangerous configurations.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    jp      = asset.data.joint_pos
    soft_lo = asset.data.soft_joint_pos_limits[..., 0]
    soft_hi = asset.data.soft_joint_pos_limits[..., 1]
    out = (
        -(jp - soft_lo * soft_ratio).clamp(max=0.0)
        + (jp - soft_hi * soft_ratio).clamp(min=0.0)
    )
    return -torch.sum(torch.abs(out), dim=-1)