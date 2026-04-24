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
#  Startup: disable ALL arm link collisions except joint6_flange (EE)
# ==============================================================================

def disable_arm_collision_except_ee(
    env: "ManagerBasedRLEnv",
    env_ids,
) -> None:
    """Disable collision on EVERY robot link except joint6_flange.

    WHY THIS IS THE CORRECT APPROACH
    ─────────────────────────────────
    The myCobot280 mesh collision shapes (convex hulls auto-generated from
    .dae meshes) have several problems in PhysX/USD:

    1. Fixed-joint links (g_base, joint1) are treated as separate static
       rigid bodies outside the articulation, so articulation-level
       self-collision filtering does NOT apply to them.

    2. Adjacent moving links (joint2/joint3, joint3/joint4 etc.) share
       overlapping convex hulls at the joint connection points. PhysX
       cannot resolve these penetrations cleanly and produces violent
       impulses that flip the arm.

    3. The mesh convex hulls are not tight — they include motor housing
       geometry that protrudes further than the kinematic envelope,
       causing false contacts during normal IK trajectories.

    SOLUTION: disable every link's CollisionAPI except joint6_flange.
    ─────────────────────────────────────────────────────────────────
    · joint6_flange keeps collision → EE physically pushes the ball ✓
    · All other links: no collision → no self-penetration impulses ✓
    · joint_pos_limits reward still prevents unsafe joint angles ✓
    · Standard practice for manipulation RL (Franka, UR5, etc.)

    The real robot has firmware self-collision checking, so removing sim
    collision does not affect sim-to-real transfer safety.

    Handles env_ids=None (startup mode) correctly.
    On startup: only env_0 is patched — USD replication propagates it
    to all 2048 envs automatically (much faster than patching every env).
    """
    import omni.usd
    from pxr import UsdPhysics, Usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    # On startup (env_ids=None): patch env_0 only — USD replication handles rest
    # On reset (env_ids=Tensor): patch the specified envs
    if env_ids is None:
        env_ids_list = [0]
    else:
        env_ids_list = env_ids.tolist()

    # Only the EE keeps its collision geometry for ball contact
    KEEP_COLLISION = {"joint6_flange"}

    total_disabled = 0

    for env_id in env_ids_list:
        robot_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot")
        if not robot_prim.IsValid():
            continue

        for prim in Usd.PrimRange(robot_prim):
            if prim.GetName() in KEEP_COLLISION:
                continue          # keep EE collision intact
            col = UsdPhysics.CollisionAPI(prim)
            if col:
                attr = col.GetCollisionEnabledAttr()
                if attr:
                    attr.Set(False)
                    total_disabled += 1

    if total_disabled == 0:
        print(
            "[WARN] disable_arm_collision_except_ee: no CollisionAPI prims found "
            "under /World/envs/env_0/Robot. Self-collision may persist."
        )
    else:
        print(
            f"[INFO] disable_arm_collision_except_ee: disabled {total_disabled} "
            f"collision prims (joint6_flange kept for ball contact)."
        )


# Keep old name as alias so existing env_cfgs that reference it still work
disable_base_link_collision = disable_arm_collision_except_ee


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
#  Hole position randomisation
# ==============================================================================

def reset_hole_position(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    hole_cfg: "SceneEntityCfg",
    ball_pos_local: tuple,
    r_nom: float,
    theta_min: float,
    theta_max: float,
) -> None:
    """Randomize hole position in a forward arc around the fixed ball position.

    Places the hole marker at distance r_nom from the ball, at a random angle
    θ ∈ [theta_min, theta_max] sampled uniformly each episode.

    Args:
        hole_cfg:       Scene entity config for the hole marker (RigidObject).
        ball_pos_local: Ball local position tuple (x, y, z).
        r_nom:          Hole distance from ball in metres.
        theta_min:      Minimum angle (rad) from +X axis around ball.
        theta_max:      Maximum angle (rad) from +X axis around ball.
    """
    hole: RigidObject = env.scene[hole_cfg.name]
    num, device = len(env_ids), env.device

    theta = torch.rand(num, device=device) * (theta_max - theta_min) + theta_min

    local = torch.zeros(num, 3, device=device)
    local[:, 0] = ball_pos_local[0] + r_nom * torch.cos(theta)
    local[:, 1] = ball_pos_local[1] + r_nom * torch.sin(theta)
    local[:, 2] = 0.002  # slight elevation for visibility

    world_pos = local + env.scene.env_origins[env_ids]
    rot = torch.zeros(num, 4, device=device)
    rot[:, 0] = 1.0  # identity quaternion (w, x, y, z)

    hole.write_root_pose_to_sim(torch.cat([world_pos, rot], dim=-1), env_ids=env_ids)


# ==============================================================================
#  Swing-ready arm reset
# ==============================================================================

# Base joint angles for swing-ready pose (user-verified in sim):
#   joint2=0° → arm in +X direction, EE just behind ball at ball height
#   joint6_to_joint5 is computed per-episode from hole angle (see below)
_SWING_JOINT_BASE = {
    "joint2_to_joint1":       0.0,
    "joint3_to_joint2":       math.radians(-100.0),
    "joint4_to_joint3":       math.radians(-38.8),
    "joint5_to_joint4":       0.0,
    "joint6output_to_joint6": 0.0,
}


def reset_arm_for_swing(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_cfg: "SceneEntityCfg",
    hole_cfg: "SceneEntityCfg",
    ball_pos_local: tuple,
) -> None:
    """Reset arm to swing-ready pose with joint6_to_joint5 aimed at hole.

    joint6_to_joint5 ≈ hole_angle (rad from +X around ball).

    This approximation is verified at φ=90° (hole in +Y from ball):
      joint6_to_joint5=90° → EE flat face points toward +Y ✓

    If the face is offset, add a calibration constant _J6_OFFSET below.
    Call reset_hole_position() BEFORE this function so hole pos is current.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    hole:  RigidObject  = env.scene[hole_cfg.name]
    joint_ids, _ = robot.find_joints(asset_cfg.joint_names)
    num, device  = len(env_ids), env.device

    # Hole world position for the requested envs
    hole_world_xy = hole.data.root_pos_w[env_ids, :2]               # (N, 2)
    origins_xy    = env.scene.env_origins[env_ids, :2]              # (N, 2)
    ball_local_xy = torch.tensor(ball_pos_local[:2], device=device).unsqueeze(0)  # (1, 2)
    ball_world_xy = origins_xy + ball_local_xy                      # (N, 2)

    to_hole_xy  = hole_world_xy - ball_world_xy                     # (N, 2)
    hole_angle  = torch.atan2(to_hole_xy[:, 1], to_hole_xy[:, 0])  # (N,) radians

    # Calibration offset — adjust if EE face direction is off (default 0)
    _J6_OFFSET: float = 0.0
    j6_target = hole_angle + _J6_OFFSET                             # (N,)

    # Build joint target tensor
    target   = torch.zeros(num, len(joint_ids), device=device)
    zero_vel = torch.zeros(num, len(joint_ids), device=device)

    for col, jid in enumerate(joint_ids):
        jname = robot.joint_names[jid]
        if jname == "joint6_to_joint5":
            target[:, col] = j6_target
        elif jname in _SWING_JOINT_BASE:
            target[:, col] = _SWING_JOINT_BASE[jname]

    robot.write_joint_state_to_sim(target, zero_vel, joint_ids=joint_ids, env_ids=env_ids)


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


# ==============================================================================
#  Scripted IK reach: teleport arm to pre-push position
# ==============================================================================

# Pre-computed joint angles (FK-verified with official Pi URDF):
#   q=(108°, 0°, -90°, 30°, 0°, 0°) → EE world ≈ (0.004, 0.193, 0.276)
#   Ball on pedestal: (0.000, 0.250, 0.276)
#   Gap: Δy=0.057m (5.7cm behind ball), Δz≈0 (same height!)
#   All joint margins ≥ 60° — comfortable, no self-collision risk
#
# This pose is written directly to the physics sim during reset.
# No IK stepping needed — arm teleports instantly.
_PREPUSH_JOINT_POS = {
    "joint2_to_joint1":       1.884956,   # 108.0°  base yaw (+Y direction)
    "joint3_to_joint2":       0.000000,   # 0.0°    shoulder NEUTRAL (q2=0)
    "joint4_to_joint3":      -1.570796,   # -90.0°  elbow extended
    "joint5_to_joint4":       0.523599,   # 30.0°   wrist tilt
    "joint6_to_joint5":       0.000000,
    "joint6output_to_joint6": 0.000000,
}


def reset_arm_to_prepush(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_cfg: "SceneEntityCfg",
) -> None:
    """Reset arm to the pre-push position using direct joint angle write.

    Architecture
    ─────────────
    Phase 1 (this function, runs during reset, deterministic):
      Home (all zeros) → Prepush q=(108°,0°,-90°,30°,0°,0°)
      EE teleports to (0.004, 0.193, 0.276) — 5.7cm behind ball, same height.
      No RL, no IK iteration needed. Just write_joint_state_to_sim.

    Phase 2 (RL policy, runs during episode):
      EE pushes ball (0, 0.25, 0.276) → hole (0, 0.40, 0.276).
      Pure horizontal +Y motion. ~5 IK steps at scale=0.04m.
      Policy only needs to learn force/direction of the push stroke.

    In SIM:  ball position = privileged physics state
    In REAL: ball position = top-down camera detection (same interface)
             The joint angles are sent via pymycobot before the RL episode.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = robot.find_joints(asset_cfg.joint_names)
    num_joints = len(joint_ids)
    num_envs   = len(env_ids)
    device     = env.device

    # Build target joint position tensor from pre-computed angles
    target_pos = torch.zeros(num_envs, num_joints, device=device)
    for col_idx, jid in enumerate(joint_ids):
        jname = robot.joint_names[jid]
        if jname in _PREPUSH_JOINT_POS:
            target_pos[:, col_idx] = _PREPUSH_JOINT_POS[jname]

    zero_vel = torch.zeros(num_envs, num_joints, device=device)

    # Teleport arm to prepush position
    robot.write_joint_state_to_sim(
        target_pos, zero_vel,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )