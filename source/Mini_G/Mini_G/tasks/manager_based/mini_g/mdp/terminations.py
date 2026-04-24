# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# SPDX-License-Identifier: BSD-3-Clause
"""
Termination conditions for the myCobot280Pi Golf Push task.

Both functions use ENV-LOCAL coordinates to work correctly across all
2048 parallel environments.  Isaac Lab tiles envs in a grid; each env has
its origin at env.scene.env_origins[i] in world space.  Computing distances
from the world origin (0,0) would always trigger termination for non-zero
env tiles.
"""

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_in_hole(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    hole_cfg: SceneEntityCfg,
    hole_radius: float = 0.046,
) -> torch.Tensor:
    """Terminate when ball XY is within hole_radius of hole centre.

    Both ball and hole positions are read in world frame from their physics
    state — correct per-env without any manual frame conversion.

    Returns boolean (N,) – True = terminate.
    """
    ball: RigidObject = env.scene[object_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]
    dist_xy = torch.linalg.norm(
        ball.data.root_pos_w[:, :2] - hole.data.root_pos_w[:, :2], dim=-1
    )
    return dist_xy < hole_radius


def ball_out_of_bounds(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    bounds_xy: float = 0.50,
    min_z:     float = -0.05,
) -> torch.Tensor:
    """Terminate when ball leaves the valid play area or falls below ground.

    bounds_xy: max XY distance from env LOCAL origin (metres).
    min_z:     world z below which ball is considered lost (< 0 = below ground).

    Returns boolean (N,) – True = terminate.
    """
    ball: RigidObject = env.scene[object_cfg.name]
    ball_local = ball.data.root_pos_w[:, :3] - env.scene.env_origins
    too_far    = torch.linalg.norm(ball_local[:, :2], dim=-1) > bounds_xy
    too_low    = ball.data.root_pos_w[:, 2] < min_z   # world z (= local z since envs are at same z)
    return too_far | too_low