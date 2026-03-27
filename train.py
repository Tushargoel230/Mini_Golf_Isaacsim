# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# Events: resets + startup collision disable + annulus ball randomisation
from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ==============================================================================
#  Startup: disable g_base and joint1 collision geometry
# ==============================================================================

def disable_base_link_collision(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Disable PhysicsCollisionAPI on g_base and joint1 across all environments.

    Must be called as a 'startup' event (env_ids=None on startup).
    """
    import omni.usd
    from pxr import UsdPhysics, Usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    # Handle startup (env_ids=None) vs reset (Tensor)
    if env_ids is None:
        env_ids_list = list(range(env.num_envs))
    else:
        env_ids_list = env_ids.tolist()

    DISABLE_LINKS = {"g_base", "joint1"}
    count = 0

    for env_id in env_ids_list:
        robot_root = f"/World/envs/env_{env_id}/Robot"
        for link_name in DISABLE_LINKS:
            for prim_path in [
                f"{robot_root}/{link_name}",
                f"{robot_root}/base_link_world/{link_name}",
            ]:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue
                for p in Usd.PrimRange(prim):
                    col = UsdPhysics.CollisionAPI(p)
                    if col:
                        attr = col.GetCollisionEnabledAttr()
                        if attr:
                            attr.Set(False)
                            count += 1

    if count == 0:
        print(
            "[WARN] disable_base_link_collision: 0 prims found. "
            "Check prim paths: /World/envs/env_N/Robot/{g_base,joint1}"
        )


# ==============================================================================
#  Ball randomisation in a reachable annulus on the table surface
# ==============================================================================

def reset_ball_in_annulus(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    r_min: float,
    r_max: float,
    theta_min: float,
    theta_max: float,
    ball_z: float,
) -> None:
    """Reset ball to a random position in a polar annulus (table surface).

    Args:
        r_min, r_max:       Radial range (m) from robot base XY.
        theta_min, theta_max: Angular range (rad) in world XY plane.
        ball_z:             Local z of ball (table surface + ball radius).
    """
    ball: RigidObject = env.scene[object_cfg.name]
    num    = len(env_ids)
    device = env.device

    # Sample radius and angle uniformly
    r     = torch.rand(num, device=device) * (r_max - r_min) + r_min
    theta = torch.rand(num, device=device) * (theta_max - theta_min) + theta_min

    local_pos = torch.zeros(num, 3, device=device)
    local_pos[:, 0] = r * torch.cos(theta)
    local_pos[:, 1] = r * torch.sin(theta)
    local_pos[:, 2] = ball_z

    # Convert to world frame
    world_pos = local_pos + env.scene.env_origins[env_ids]

    new_rot  = torch.zeros(num, 4, device=device); new_rot[:, 0] = 1.0
    zero_vel = torch.zeros(num, 6, device=device)

    ball.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
    ball.write_root_pose_to_sim(
        torch.cat([world_pos, new_rot], dim=-1), env_ids=env_ids
    )
    ball.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)


# ==============================================================================
#  Standard resets
# ==============================================================================

def reset_joints_by_offset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    position_range: tuple[float, float] = (-0.05, 0.05),
    velocity_range: tuple[float, float] = (-0.01, 0.01),
) -> None:
    """Reset joint positions to default ± uniform noise."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids  = robot.find_joints(asset_cfg.joint_names)[0]
    num        = len(env_ids)
    num_joints = len(joint_ids)
    device     = env.device

    default_pos = robot.data.default_joint_pos[env_ids][:, joint_ids]
    default_vel = robot.data.default_joint_vel[env_ids][:, joint_ids]

    pos_noise = (
        torch.rand(num, num_joints, device=device)
        * (position_range[1] - position_range[0]) + position_range[0]
    )
    vel_noise = (
        torch.rand(num, num_joints, device=device)
        * (velocity_range[1] - velocity_range[0]) + velocity_range[0]
    )
    robot.write_joint_state_to_sim(
        default_pos + pos_noise, default_vel + vel_noise,
        joint_ids=joint_ids, env_ids=env_ids,
    )


from isaaclab.envs.mdp.events import reset_scene_to_default  # noqa: F401, E402


# ==============================================================================
#  Viewport setup  –  opens a second Isaac Sim panel showing the top-down camera
# ==============================================================================

def open_camera_viewport(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Startup event: open a second viewport showing the top-down camera feed.

    This runs ONCE after scene creation (startup mode, env_ids=None).

    What it does
    ─────────────
    Isaac Sim's GUI supports multiple viewport panels. This function:
    1. Opens a second viewport window titled "Top Camera View"
    2. Points it at env_0's TopCamera prim
    3. Sizes it to 640×480 (matches the real AI Kit camera resolution)

    The main viewport keeps the default perspective view (useful for debugging).
    The camera viewport shows exactly what the real USB camera will see.

    Headless mode
    ─────────────
    When running with --headless the viewport windows cannot be created.
    The function detects this and exits silently – training is unaffected.

    How to use in GUI
    ─────────────────
    Run WITHOUT --headless:
      python scripts/skrl/train.py --task Mini-G-Golf-v0
    You will see two viewport panels:
      · Left:  Perspective (main viewer)
      · Right: Top Camera View (640×480, 25 Hz)
    """
    # Only works in GUI mode (not headless)
    try:
        import omni.kit.app
        app = omni.kit.app.get_app()
        if not hasattr(app, "_app") and app is None:
            return  # no app running
    except Exception:
        return

    try:
        from omni.kit.viewport.window import ViewportWindow
        import omni.usd

        # Camera prim in env_0 (all envs render identically; we just show env_0)
        cam_prim_path = "/World/envs/env_0/TopCamera"

        # Check camera prim exists before creating viewport
        stage = omni.usd.get_context().get_stage()
        if stage is None or not stage.GetPrimAtPath(cam_prim_path).IsValid():
            print(
                f"[WARN] open_camera_viewport: prim not found at {cam_prim_path}. "
                "Viewport not created."
            )
            return

        # Create a named viewport window
        vp_window = ViewportWindow(
            "Top Camera View",
            width=640,
            height=480,
        )
        # Point it at the top-down camera
        vp_window.viewport_api.camera_path = cam_prim_path

        print(
            f"[INFO] Camera viewport opened → '{cam_prim_path}'\n"
            "       Dock the 'Top Camera View' panel anywhere in the Isaac Sim UI."
        )

    except ImportError:
        # ViewportWindow not available (older Isaac Sim or headless kit)
        _open_viewport_fallback()
    except Exception as e:
        print(f"[WARN] open_camera_viewport: {e}")


def _open_viewport_fallback() -> None:
    """Fallback: use omni.kit.viewport.utility if ViewportWindow unavailable."""
    try:
        import omni.kit.viewport.utility as vp_util

        # Get or create a second viewport
        existing = vp_util.get_viewport_from_window_name("Top Camera View")
        if existing is None:
            vp_util.create_viewport_window("Top Camera View", width=640, height=480)
            viewport = vp_util.get_viewport_from_window_name("Top Camera View")
        else:
            viewport = existing

        if viewport:
            viewport.camera_path = "/World/envs/env_0/TopCamera"
            print("[INFO] Camera viewport opened via viewport.utility fallback.")
    except Exception as e:
        print(f"[WARN] Viewport fallback also failed: {e}")