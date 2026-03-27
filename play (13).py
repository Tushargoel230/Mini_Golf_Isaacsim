# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Sim-To-Real: myCobot280Pi Golf Push Task
# ──────────────────────────────────────────────────────────────────────────────
# TASK FRAMING  – "detect → reach → push"
#   In SIM:  ball position = privileged state (perfect "camera")
#   In REAL: ball position = top-down USB camera + colour detection
#            (Elephant Robotics AI Kit camera, mounted on robot base)
#   IK:      DifferentialIK converts Δ(x,y,z) EE commands → joint angles
#   Policy:  learns PUSH DIRECTION + VELOCITY (not hardcoded reaching)
#
# EPISODE RESET BEHAVIOUR
# ──────────────────────────────────────────────────────────────────────────────
#   Ball:   always resets to BALL_INIT_POS (fixed position, no randomisation)
#           reset_scene_to_default handles this automatically via init_state.
#   Robot:  always resets to the EXACT HOME POSE defined in joint_pos below.
#           position_range=(0,0) velocity_range=(0,0) → no noise added.
#
# REWARD STRUCTURE
#   r1  EE approaches ball              dense, 1/(1+k·d)
#   r2  Ball velocity toward hole       push force & direction
#   r3  Ball distance to hole           dense guidance
#   r4  Hole-in-one                     sparse terminal +200
#   p1  Action smoothness               penalise jerk
#   p2  Joint limits                    safety
# ==============================================================================

from __future__ import annotations
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from . import mdp


# ==============================================================================
#  Hardware constants  (myCobot280Pi manual + URDF)
# ==============================================================================

_JOINT_NAMES: list[str] = [
    "joint2_to_joint1",        # J1 – base yaw        ±168°
    "joint3_to_joint2",        # J2 – shoulder pitch  ±135°
    "joint4_to_joint3",        # J3 – elbow           ±150°
    "joint5_to_joint4",        # J4 – wrist1          ±145°
    "joint6_to_joint5",        # J5 – wrist2          ±165°
    "joint6output_to_joint6",  # J6 – flange          ±180°
]

_EE_BODY  = "joint6_flange"
_USD_PATH = (
    "/home/goel/code-goel/MyCobot280_Pi/"
    "mycobot_description/urdf/mycobotgg/mycobotgg.usd"
)

# ==============================================================================
#  Scene geometry  (user-defined positions)
# ==============================================================================

BASE_Z:          float = 0.030    # robot base world z
BALL_RADIUS:     float = 0.021
HOLE_RADIUS:     float = 0.042
APPROACH_HEIGHT: float = 0.08     # IK approach height above ball

_BALL_R_NOM:     float = 0.25
_BALL_THETA_NOM: float = math.radians(90.0)

BALL_INIT_POS: tuple = (
    _BALL_R_NOM * math.cos(_BALL_THETA_NOM),   # = 0.0
    _BALL_R_NOM * math.sin(_BALL_THETA_NOM),   # = 0.21
    BALL_RADIUS,                                # = 0.021m  ON the ground
)

_HOLE_R: float = _BALL_R_NOM + 0.15
HOLE_POS: tuple = (
    _HOLE_R * math.cos(_BALL_THETA_NOM),   # = 0.0
    _HOLE_R * math.sin(_BALL_THETA_NOM),   # = 0.30
    0.002,
)


# ==============================================================================
#  myCobot280Pi ArticulationCfg
# ==============================================================================

_MYCOBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
            disable_gravity=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, BASE_Z),
        rot=(1.0, 0.0, 0.0, 0.0),
        # ── FIXED HOME POSE  ──────────────────────────────────────────────────
        # This pose is restored EXACTLY every episode (position_range=(0,0)).
        # To change the home pose, edit these angles — the robot will always
        # return to exactly these values at the start of every new episode.
        joint_pos={
            "joint2_to_joint1":       math.radians(0.0),
            "joint3_to_joint2":       math.radians(0.0),
            "joint4_to_joint3":       math.radians(0.0),
            "joint5_to_joint4":       math.radians(0.0),
            "joint6_to_joint5":       0.0,
            "joint6output_to_joint6": 0.0,
        },
        joint_vel={name: 0.0 for name in _JOINT_NAMES},
    ),
    actuators={
        "mycobot_arm_joints": ImplicitActuatorCfg(
            joint_names_expr=_JOINT_NAMES,
            effort_limit_sim=10.0,
            velocity_limit_sim=3.5,
            stiffness=800.0,
            damping=50.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)


# ==============================================================================
#  Golf Ball
# ==============================================================================

GOLF_BALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/GolfBall",
    spawn=sim_utils.SphereCfg(
        radius=BALL_RADIUS,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            max_angular_velocity=100.0,
            max_linear_velocity=2.0,
            max_depenetration_velocity=0.1,
            disable_gravity=False,
            linear_damping=50.0,
            angular_damping=50.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.046),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.001,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.90,
            dynamic_friction=0.80,
            restitution=0.0,
            friction_combine_mode="max",
            restitution_combine_mode="min",
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.45, 0.0),
            roughness=0.4,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=BALL_INIT_POS,
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
)


# ==============================================================================
#  Hole marker
# ==============================================================================

HOLE_MARKER_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/HoleMarker",
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(HOLE_POS[0], HOLE_POS[1], 0.0),
    ),
    spawn=sim_utils.CylinderCfg(
        radius=HOLE_RADIUS,
        height=0.003,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.05, 0.05, 0.05),
            roughness=1.0,
        ),
    ),
)


# ==============================================================================
#  Top-down camera  (for GUI viewport only, does NOT feed the policy)
# ==============================================================================

_CAM_HEIGHT = 0.50
TOP_DOWN_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/TopCamera",
    update_period=0.04,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=500.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 10.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, 0.30, _CAM_HEIGHT),
        rot=(0.0, 0.7071, 0.7071, 0.0),
        convention="ros",
    ),
)


# ==============================================================================
#  Scene
# ==============================================================================

@configclass
class MiniGolfSceneCfg(InteractiveSceneCfg):
    """Headless training scene — no camera."""

    ground     = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    sky_light  = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)),
    )
    robot:       ArticulationCfg = _MYCOBOT_CFG
    golf_ball:   RigidObjectCfg  = GOLF_BALL_CFG
    hole_marker: AssetBaseCfg    = HOLE_MARKER_CFG


@configclass
class MiniGolfSceneCfg_Visual(MiniGolfSceneCfg):
    """GUI scene — adds top-down camera viewport (requires --enable_cameras)."""
    top_camera: CameraCfg = TOP_DOWN_CAMERA_CFG


# ==============================================================================
#  Observations  (30-dim)
#  joint_pos(6) + joint_vel(6) + ee_pos(3) + ball_pos(3) + ball_vel(3)
#  + ee_to_ball(3) + ball_to_hole(3) + ee_to_approach(3) = 30
# ==============================================================================

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        ee_position = ObsTerm(
            func=mdp.body_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[_EE_BODY])},
        )
        ball_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("golf_ball")},
        )
        ball_velocity = ObsTerm(
            func=mdp.object_lin_vel_w,
            params={"object_cfg": SceneEntityCfg("golf_ball")},
            scale=0.5,
        )
        ee_to_ball = ObsTerm(
            func=mdp.ee_to_object_vector,
            params={
                "ee_cfg":     SceneEntityCfg("robot", body_names=[_EE_BODY]),
                "object_cfg": SceneEntityCfg("golf_ball"),
            },
        )
        ball_to_hole = ObsTerm(
            func=mdp.ball_to_hole_vector,
            params={
                "object_cfg": SceneEntityCfg("golf_ball"),
                "hole_pos":   HOLE_POS,
            },
        )
        ee_to_approach_target = ObsTerm(
            func=mdp.ee_to_approach_target,
            params={
                "ee_cfg":          SceneEntityCfg("robot", body_names=[_EE_BODY]),
                "object_cfg":      SceneEntityCfg("golf_ball"),
                "approach_height": APPROACH_HEIGHT,
            },
        )

        def __post_init__(self):
            self.enable_corruption  = False
            self.concatenate_terms  = True

    policy: PolicyCfg = PolicyCfg()


# ==============================================================================
#  Actions  –  Differential IK position deltas
# ==============================================================================

@configclass
class ActionsCfg:
    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=_JOINT_NAMES,
        body_name=_EE_BODY,
        controller=DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": 0.01},
        ),
        scale=0.04,
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.012),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# ==============================================================================
#  Events
# ==============================================================================

@configclass
class EventCfg:
    # ── Startup ───────────────────────────────────────────────────────────────
    disable_base_collision = EventTerm(
        func=mdp.disable_base_link_collision,
        mode="startup",
    )
    open_viewport = EventTerm(
        func=mdp.open_camera_viewport,
        mode="startup",
    )

    # ── Reset ─────────────────────────────────────────────────────────────────
    # 1. Full physics reset: restores ball to BALL_INIT_POS, robot to home pose.
    #    This is all that's needed because init_state holds the fixed positions.
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    # 2. Robot joints reset to EXACT home pose every episode.
    #    position_range=(0.0, 0.0) → zero offset added → always exact home pose.
    #    velocity_range=(0.0, 0.0) → zero velocity at episode start.
    #
    #    To change the home pose: edit joint_pos in _MYCOBOT_CFG.init_state above.
    #    The robot will start from those exact angles every single episode.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg":      SceneEntityCfg("robot", joint_names=_JOINT_NAMES),
            "position_range": (0.0, 0.0),   # ← exact home pose, no noise
            "velocity_range": (0.0, 0.0),   # ← zero velocity
        },
    )

    # NOTE: reset_ball_random is intentionally REMOVED.
    # Ball always resets to BALL_INIT_POS via reset_scene_to_default above.


# ==============================================================================
#  Rewards
# ==============================================================================

@configclass
class RewardsCfg:
    ee_approach_target = RewTerm(
        func=mdp.ee_approach_ball,
        weight=3.0,
        params={
            "ee_cfg":     SceneEntityCfg("robot", body_names=[_EE_BODY]),
            "object_cfg": SceneEntityCfg("golf_ball"),
            "k": 8.0,
        },
    )
    ee_approach_ball = RewTerm(
        func=mdp.ee_approach_ball,
        weight=2.0,
        params={
            "ee_cfg":     SceneEntityCfg("robot", body_names=[_EE_BODY]),
            "object_cfg": SceneEntityCfg("golf_ball"),
            "k": 10.0,
        },
    )
    ee_height_match = RewTerm(
        func=mdp.ee_height_match,
        weight=1.5,
        params={
            "ee_cfg":     SceneEntityCfg("robot", body_names=[_EE_BODY]),
            "object_cfg": SceneEntityCfg("golf_ball"),
        },
    )
    ball_velocity_toward_hole = RewTerm(
        func=mdp.ball_velocity_toward_hole,
        weight=8.0,
        params={
            "object_cfg": SceneEntityCfg("golf_ball"),
            "hole_pos":   HOLE_POS,
            "vel_scale":  5.0,
        },
    )
    ball_to_hole = RewTerm(
        func=mdp.ball_to_hole_reward,
        weight=5.0,
        params={
            "object_cfg": SceneEntityCfg("golf_ball"),
            "hole_pos":   HOLE_POS,
            "k": 6.0,
        },
    )
    hole_in_one = RewTerm(
        func=mdp.hole_in_one_bonus,
        weight=200.0,
        params={
            "object_cfg":  SceneEntityCfg("golf_ball"),
            "hole_pos":    HOLE_POS,
            "hole_radius": HOLE_RADIUS,
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg":  SceneEntityCfg("robot", joint_names=_JOINT_NAMES),
            "soft_ratio": 0.95,
        },
    )


# ==============================================================================
#  Terminations
# ==============================================================================

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    ball_in_hole = DoneTerm(
        func=mdp.ball_in_hole,
        params={
            "object_cfg":  SceneEntityCfg("golf_ball"),
            "hole_pos":    HOLE_POS,
            "hole_radius": HOLE_RADIUS * 1.1,
        },
    )
    ball_out_of_bounds = DoneTerm(
        func=mdp.ball_out_of_bounds,
        params={
            "object_cfg": SceneEntityCfg("golf_ball"),
            "bounds_xy":  0.50,
            "min_z":      -0.05,
        },
    )


# ==============================================================================
#  Environment configs
# ==============================================================================

@configclass
class MiniGolfEnvCfg(ManagerBasedRLEnvCfg):
    """Headless training — ball fixed, robot always starts from exact home pose.

    Episode reset:
      - Ball:  always at BALL_INIT_POS = (0.0, 0.21, 0.021)
      - Robot: always at joint_pos defined in _MYCOBOT_CFG.init_state
               (45°, 120°, 135°, 30°, 0°, 0°)

    To change the home pose: edit joint_pos in _MYCOBOT_CFG.init_state.
    To enable camera viewport: use Mini-G-Golf-Visual-v0 with --enable_cameras.
    """

    scene: MiniGolfSceneCfg   = MiniGolfSceneCfg(num_envs=2048, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg           = ActionsCfg()
    rewards: RewardsCfg           = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg              = EventCfg()

    def __post_init__(self):
        self.sim.dt           = 0.02
        self.decimation       = 2
        self.episode_length_s = 6.0
        self.viewer.eye       = (0.60, -0.20, 0.55)
        self.viewer.lookat    = (0.05, 0.15, 0.05)
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",
            restitution_combine_mode="min",
            static_friction=0.80,
            dynamic_friction=0.70,
            restitution=0.0,
        )


@configclass
class MiniGolfEnvCfg_Visual(MiniGolfEnvCfg):
    """GUI variant with top-down camera. Requires --enable_cameras."""
    scene: MiniGolfSceneCfg_Visual = MiniGolfSceneCfg_Visual(
        num_envs=2048, env_spacing=2.0
    )


@configclass
class MiniGolfEnvCfg_PLAY(MiniGolfEnvCfg):
    """Evaluation: 4 envs, deterministic, longer episodes."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs    = 4
        self.scene.env_spacing = 2.5
        self.episode_length_s  = 12.0