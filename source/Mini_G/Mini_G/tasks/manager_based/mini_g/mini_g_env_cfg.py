# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Sim-To-Real: myCobot280Pi Golf Push — TASK-SPACE RL
# ──────────────────────────────────────────────────────────────────────────────
#  PROFESSOR'S DESIGN: learn EE Cartesian trajectory, not joint angles
# ──────────────────────────────────────────────────────────────────────────────
#
#  CONCEPT
#  ────────
#  Swing-based push: EE resets BEHIND the ball (opposite hole direction) at
#  ball height, then swings horizontally through the ball toward the hole.
#  Like a billiards cue shot / golf putt.
#
#    reset: hole randomized → arm set at joint2=0° (arm in +X, EE behind ball)
#           joint6_to_joint5 = hole_angle (EE flat face toward hole)
#         │
#         │  RL episode starts. Policy outputs Δ(x,y,z) EE deltas.
#         │  DifferentialIK converts to joint targets (mainly joint2 swing).
#         │  Policy NEVER sees joint angles — only task-space quantities.
#         ▼
#    EE swings through ball → ball rolls toward randomized hole → HOLE
#
#  WHY TASK-SPACE?
#  ───────────────
#  · Policy input:  EE position, ball position, vectors (all Cartesian)
#  · Policy output: Δ(x,y,z) EE displacement (Cartesian)
#  · Joint angles:  handled transparently by DifferentialIK (not observed)
#  · Sim-to-real: same Cartesian commands → pymycobot.send_coords() on real robot
#  · Observations are 18-dim: removed joint_pos(6) + joint_vel(6)
#
#  GEOMETRY (user-verified: reset pose q=(0°,-100°,-38.8°,0°,φ,0°))
#  ─────────────────────────────────────────────────────────────────
#    Ball:       (0.21868, 0.04051, 0.021)  ← fixed each episode
#    Reset EE:   behind ball in +X direction (joint2=0°)
#    Hole:       0.15m from ball, angle φ ∈ [0°, 180°] randomized each episode
#    joint6_to_joint5 ≈ φ: orients EE flat face toward hole
#    Swing:      joint2 rotation sweeps EE horizontally through ball
#
#  OBSERVATION SPACE (18-dim — pure task space)
#  ─────────────────────────────────────────────
#    ee_pos          (3)  EE Cartesian world position
#    ball_pos        (3)  ball position relative to robot root
#    ball_vel        (3)  ball velocity × 0.5 (contact/push signal)
#    ee_to_ball      (3)  vector EE → ball centre
#    ball_to_hole    (3)  vector ball → hole (dynamic, world-frame, per-env)
#    ee_to_backswing (3)  vector EE → backswing pos (behind ball, opposite hole)
#    ────────────────
#    total          18
#
#  NO joint_pos, NO joint_vel — policy operates entirely in task space.
# ==============================================================================

from __future__ import annotations
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg  # noqa: F401
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
#  Hardware constants
# ==============================================================================

_JOINT_NAMES: list[str] = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
]
_EE_BODY  = "joint6_flange"
_USD_PATH = (
    "/home/goel/code-goel/MyCobot280_Pi/"
    "mycobot_description/urdf/mycobotgg/mycobotgg.usd"
)


# ==============================================================================
#  HOME/RESET POSE  (user-verified in sim: EE just behind ball at ball height)
#  ─────────────────────────────────────────────────────────────────────────────
#  q=(0°, -100°, -38.8°, 0°, 90°, 0°)  with joint2=0° → arm in +X direction
#  EE positioned behind ball at (0.21868, 0.04051) — flat face toward +Y (hole)
#  joint6_to_joint5 is overridden per-episode by reset_arm_for_swing() based on hole angle
# ==============================================================================

_HOME_JOINT_POS = {
    "joint2_to_joint1":       math.radians(  0.0),   # arm in +X, EE behind ball
    "joint3_to_joint2":       math.radians(-100.0),
    "joint4_to_joint3":       math.radians( -38.8),
    "joint5_to_joint4":       math.radians(   0.0),
    "joint6_to_joint5":       math.radians(  90.0),  # nominal: EE faces +Y (φ=90°)
    "joint6output_to_joint6": 0.0,
}


# ==============================================================================
#  Scene geometry  (YOUR positions — unchanged)
# ==============================================================================

BASE_Z:      float = 0.030
BALL_RADIUS: float = 0.021
HOLE_RADIUS: float = 0.042

# Ball fixed at user-verified position (EE at joint2=0° sits just behind ball)
BALL_INIT_POS: tuple = (0.22868, 0.0351, BALL_RADIUS)

# Swing / hole geometry
BACKSWING_DIST: float = 0.07            # 7 cm behind ball for ee_to_backswing obs
HOLE_DIST:      float = 0.28            # 28 cm — further hole, more roll required
# 30° arc centred on swing direction (+Y from ball, 90° from +X):
# Narrower arc → more consistent training target
HOLE_THETA_MIN: float = math.radians( 75.0)
HOLE_THETA_MAX: float = math.radians(105.0)

# Nominal hole position for asset init_state (overwritten at first reset)
_HOLE_INIT_X: float = BALL_INIT_POS[0] + HOLE_DIST * math.cos(math.radians(90.0))
_HOLE_INIT_Y: float = BALL_INIT_POS[1] + HOLE_DIST * math.sin(math.radians(90.0))


# ==============================================================================
#  Robot
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
        joint_pos=_HOME_JOINT_POS,
        joint_vel={name: 0.0 for name in _JOINT_NAMES},
    ),
    actuators={
        "mycobot_arm_joints": ImplicitActuatorCfg(
            joint_names_expr=_JOINT_NAMES,
            effort_limit_sim=3.0,    # realistic servo torque ~1–3 N·m on myCobot280
            velocity_limit_sim=3.0,  # matches real motor at speed=80 (~137°/s ≈ 2.4 rad/s)
            stiffness=40.0,          # slower position tracking → mimics servo firmware ramp
            damping=5.6,             # keeps ζ ≈ 0.44 consistent: 5.6/(2√40) ≈ 0.44
        ),
    },
    soft_joint_pos_limit_factor=0.90,  # URDF J5 ±165°; 90% → ±148.5° < real 160° cap
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
            max_linear_velocity=3.0,
            max_depenetration_velocity=0.1,
            disable_gravity=False,
            linear_damping=0.0,#5,   # low drag — ball rolls freely on hard surface
            angular_damping=0.001,#5,  # low spin decay
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.046),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.001,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.35,
            dynamic_friction=0.02,
            restitution=0.15,
            friction_combine_mode="multiply",
            restitution_combine_mode="min",
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0., 0.50),
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

HOLE_MARKER_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/HoleMarker",
    spawn=sim_utils.CylinderCfg(
        radius=HOLE_RADIUS,
        height=0.003,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,   # repositioned via write_root_pose, not physics
            disable_gravity=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.05, 0.05, 0.05),
            roughness=1.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(_HOLE_INIT_X, _HOLE_INIT_Y, 0.002)),
)


# ==============================================================================
#  Camera  (GUI viewport only)
# ==============================================================================

TOP_DOWN_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/TopCamera",
    update_period=0.04,
    height=480, width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=500.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 10.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.22, 0.14, 1.20),   # centred on ball/hole area, 1.2 m height
        rot=(0.0, 0.7071, 0.7071, 0.0),
        convention="ros",
    ),
)


# ==============================================================================
#  Scene
# ==============================================================================

@configclass
class MiniGolfSceneCfg(InteractiveSceneCfg):
    ground      = AssetBaseCfg(prim_path="/World/ground",
                                spawn=sim_utils.GroundPlaneCfg())
    sky_light   = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)),
    )
    robot:       ArticulationCfg = _MYCOBOT_CFG
    golf_ball:   RigidObjectCfg  = GOLF_BALL_CFG
    hole_marker: RigidObjectCfg  = HOLE_MARKER_CFG


@configclass
class MiniGolfSceneCfg_Visual(MiniGolfSceneCfg):
    top_camera: CameraCfg = TOP_DOWN_CAMERA_CFG


# ==============================================================================
#  Observations  (18-dim — PURE TASK SPACE, no joint angles)
#  ─────────────────────────────────────────────────────────────────────────────
#  Removing joint_pos(6) + joint_vel(6) from the observation:
#  · The policy operates entirely in Cartesian space
#  · DifferentialIK handles the joint-space mapping transparently
#  · Observations are all physically meaningful task-space quantities
#  · Generalises better to real robot (no joint-angle domain gap)
#
#  Dim  Term              Notes
#   3   ee_pos            EE Cartesian world position
#   3   ball_pos          ball position relative to robot root
#   3   ball_vel × 0.5    ball velocity (fires after contact)
#   3   ee_to_ball        EE→ball vector (contact guidance)
#   3   ball_to_hole      ball→hole vector (dynamic, world-frame, per-env)
#   3   ee_to_backswing   EE→backswing position (behind ball, opposite hole)
#  ──   ──────────────────
#  18   total
# ==============================================================================

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # ── Task-space EE position ────────────────────────────────────────────
        ee_position = ObsTerm(
            func=mdp.body_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[_EE_BODY])},
        )
        # ── Ball state ────────────────────────────────────────────────────────
        ball_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("golf_ball")},
        )
        ball_velocity = ObsTerm(
            func=mdp.object_lin_vel_w,
            params={"object_cfg": SceneEntityCfg("golf_ball")},
            scale=0.5,
        )
        # ── Cartesian error vectors (encode task structure) ───────────────────
        # These vectors directly encode what the EE needs to do:
        # ee_to_ball → 0  means EE has reached the ball (contact)
        # ball_to_hole     tells the policy which direction to push
        # ee_to_approach → 0  means EE is correctly positioned for descent
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
                "hole_cfg":   SceneEntityCfg("hole_marker"),
            },
        )
        ee_to_backswing = ObsTerm(
            func=mdp.ee_to_backswing,
            params={
                "ee_cfg":         SceneEntityCfg("robot", body_names=[_EE_BODY]),
                "object_cfg":     SceneEntityCfg("golf_ball"),
                "hole_cfg":       SceneEntityCfg("hole_marker"),
                "backswing_dist": BACKSWING_DIST,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ==============================================================================
#  Actions — Cartesian EE position deltas via DifferentialIK
#  ─────────────────────────────────────────────────────────────────────────────
#  Policy outputs Δ(x,y,z) in Cartesian space (3-dim).
#  DifferentialIK (DLS Jacobian inverse) maps these to 6 joint targets.
#  The policy NEVER reasons about joint angles — it plans in task space.
#  scale=0.04m/step × 25Hz = 1.0 m/s maximum EE speed.
# ==============================================================================

@configclass
class ActionsCfg:
    """Task-space action: policy outputs Δ(x,y,z) EE position deltas.

    DifferentialIK maps these Cartesian commands to joint position targets
    internally — the policy never reasons about joint angles.

    body_offset=(0,0,0): joint6_flange origin is exactly at the outer face
    of the cylindrical flange — the physical contact tip. No offset needed.
    """
    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=_JOINT_NAMES,
        body_name=_EE_BODY,
        controller=DifferentialIKControllerCfg(
            command_type="pose",  # 3-dim Δ(x,y,z) — "pose" breaks network output shape
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": 0.01},
        ),
        scale=0.04,   # 0.04 m/step × 25 Hz = 1.0 m/s max EE speed
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),   # flange centre = contact point, no offset
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# ==============================================================================
#  Events
# ==============================================================================

@configclass
class EventCfg:
    # ── Startup ───────────────────────────────────────────────────────────────
    # Disable ALL arm link collisions except joint6_flange (EE contact point).
    disable_arm_collision = EventTerm(
        func=mdp.disable_arm_collision_except_ee,
        mode="startup",
    )
    open_viewport = EventTerm(
        func=mdp.open_camera_viewport,
        mode="startup",
    )

    # ── Reset  (order matters: hole first → arm reads hole angle) ────────────
    # 1. Restore all assets to default poses
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )
    # 2. Randomize hole position in a forward arc around the ball
    reset_hole = EventTerm(
        func=mdp.reset_hole_position,
        mode="reset",
        params={
            "hole_cfg":       SceneEntityCfg("hole_marker"),
            "ball_pos_local": BALL_INIT_POS,
            "r_nom":          HOLE_DIST,
            "theta_min":      HOLE_THETA_MIN,
            "theta_max":      HOLE_THETA_MAX,
        },
    )
    # 3. Set arm to swing-ready pose; joint6_to_joint5 computed from hole angle
    reset_arm = EventTerm(
        func=mdp.reset_arm_for_swing,
        mode="reset",
        params={
            "asset_cfg":      SceneEntityCfg("robot", joint_names=_JOINT_NAMES),
            "hole_cfg":       SceneEntityCfg("hole_marker"),
            "ball_pos_local": BALL_INIT_POS,
        },
    )


# ==============================================================================
#  Rewards  (task-space, ball-centric)
#  ─────────────────────────────────────────────────────────────────────────────
#  Professor's design: EE reaching is handled by IK — RL only learns the push.
#
#  EE rewards are REMOVED or minimal:
#    - ee_approach_ball removed: ball is placed just below home EE, IK descends
#    - ee_height_match kept at w=0.3 only: tiny nudge to prevent arm staying
#      high, but NOT a primary signal (IK already handles height matching)
#
#  Ball rewards dominate:
#    ball_velocity_toward_hole  w=12.0  PRIMARY: push force + direction
#    ball_to_hole               w= 6.0  SECONDARY: displacement progress
#    hole_in_one                w=200   TERMINAL: success bonus
#
#  The policy learns: "what Cartesian path should the EE follow to impart
#  the right velocity to the ball in the direction of the hole?"
# ==============================================================================

@configclass
class RewardsCfg:
    # ── EE height match: critical for horizontal swing at ball height ─────────
    # Weight 1.5: stronger signal — EE must swing at ball height, not above it.
    ee_height_match = RewTerm(
        func=mdp.ee_height_match,
        weight=1.5,
        params={
            "ee_cfg":     SceneEntityCfg("robot", body_names=[_EE_BODY]),
            "object_cfg": SceneEntityCfg("golf_ball"),
        },
    )

    # ── PRIMARY: ball velocity toward hole ────────────────────────────────────
    # tanh(5 · max(0, v_ball · ĥ_hole))
    # Rewards push force AND direction; hole_cfg provides per-env world position.
    # Fires only after EE contacts ball — zero for stationary ball.
    ball_velocity_toward_hole = RewTerm(
        func=mdp.ball_velocity_toward_hole,
        weight=12.0,
        params={
            "object_cfg": SceneEntityCfg("golf_ball"),
            "hole_cfg":   SceneEntityCfg("hole_marker"),
            "vel_scale":  1.0,
        },
    )

    # ── SECONDARY: ball displacement toward hole ──────────────────────────────
    ball_to_hole = RewTerm(
        func=mdp.ball_to_hole_reward,
        weight=6.0,
        params={
            "object_cfg": SceneEntityCfg("golf_ball"),
            "hole_cfg":   SceneEntityCfg("hole_marker"),
            "k": 6.0,
        },
    )

    # ── TERMINAL: hole-in-one ─────────────────────────────────────────────────
    hole_in_one = RewTerm(
        func=mdp.hole_in_one_bonus,
        weight=200.0,
        params={
            "object_cfg":  SceneEntityCfg("golf_ball"),
            "hole_cfg":    SceneEntityCfg("hole_marker"),
            "hole_radius": HOLE_RADIUS,
        },
    )

    # ── ONE-TOUCH ENFORCEMENT ─────────────────────────────────────────────────
    # Reward EE approaching ball ONLY while ball is stationary (pre-hit).
    # Once ball is moving, this term gives zero — no reward for chasing the ball.
    first_contact_approach = RewTerm(
        func=mdp.first_contact_approach,
        weight=3.0,
        params={
            "ee_cfg":          SceneEntityCfg("robot", body_names=[_EE_BODY]),
            "object_cfg":      SceneEntityCfg("golf_ball"),
            "vel_threshold":   0.05,   # m/s — ball considered "hit" above this
            "k":               10.0,
        },
    )
    # Penalize EE being near ball when ball is already in motion (second hit).
    second_contact_penalty = RewTerm(
        func=mdp.second_contact_penalty,
        weight=8.0,
        params={
            "ee_cfg":         SceneEntityCfg("robot", body_names=[_EE_BODY]),
            "object_cfg":     SceneEntityCfg("golf_ball"),
            "vel_threshold":  0.05,    # same threshold as above
            "contact_dist":   0.06,    # within 6 cm = "contact zone"
        },
    )

    # ── PENALTIES ─────────────────────────────────────────────────────────────
    # Penalizes large action changes → policy prefers one decisive swing
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=0.01)
    # Safety: keep joints away from limits
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
            "hole_cfg":    SceneEntityCfg("hole_marker"),
            "hole_radius": HOLE_RADIUS * 1.1,
        },
    )
    ball_out_of_bounds = DoneTerm(
        func=mdp.ball_out_of_bounds,
        params={
            "object_cfg": SceneEntityCfg("golf_ball"),
            "bounds_xy":  0.60,
            "min_z":      -0.05,
        },
    )


# ==============================================================================
#  Environment configs
# ==============================================================================

@configclass
class MiniGolfEnvCfg(ManagerBasedRLEnvCfg):
    """myCobot280Pi Golf — Swing-Based Task-Space RL.

    The policy learns a Cartesian EE swing trajectory to push the ball into a
    randomized hole. DifferentialIK handles joint mapping; policy reasons in task space.
    Observations are 18-dim (no joint angles).

    Episode flow:
      reset → hole randomized in forward arc → arm set behind ball (joint2=0°,
              joint6_to_joint5=hole_angle) → RL: swing through ball → hole
    Ball: (0.21868, 0.04051, 0.021m)  Hole: 0.15m from ball, angle 0°–180°

    Train:  python scripts/skrl/train.py --task Mini-G-Golf-v0 --headless
    Visual: python scripts/skrl/train.py --task Mini-G-Golf-Visual-v0 --enable_cameras
    Eval:   python scripts/skrl/play.py  --task Mini-G-Golf-Play-v0 --checkpoint ...
    """
    scene:        MiniGolfSceneCfg = MiniGolfSceneCfg(num_envs=2048, env_spacing=2.0)
    observations: ObservationsCfg  = ObservationsCfg()
    actions:      ActionsCfg       = ActionsCfg()
    rewards:      RewardsCfg       = RewardsCfg()
    terminations: TerminationsCfg  = TerminationsCfg()
    events:       EventCfg         = EventCfg()

    def __post_init__(self):
        self.sim.dt           = 0.02
        self.decimation       = 2
        self.episode_length_s = 6.0   # 150 steps at 25Hz

        self.viewer.eye    = (0.55,  -0.20, 0.70)
        self.viewer.lookat = (0.22,   0.14, 0.02)

        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",  # was "max" — "max" overrides ball's "multiply" mode (PhysX priority), causing effective friction=0.80 instead of 0.35×0.40=0.14
            restitution_combine_mode="min",
            static_friction=0.40,
            dynamic_friction=0.40,
            restitution=0.0,
        )


@configclass
class MiniGolfEnvCfg_Visual(MiniGolfEnvCfg):
    scene: MiniGolfSceneCfg_Visual = MiniGolfSceneCfg_Visual(
        num_envs=2048, env_spacing=2.0
    )


@configclass
class MiniGolfEnvCfg_PLAY(MiniGolfEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs    = 4
        self.scene.env_spacing = 2.5
        self.episode_length_s  = 12.0