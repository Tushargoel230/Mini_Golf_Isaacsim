#!/usr/bin/env python3
# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# SPDX-License-Identifier: BSD-3-Clause
"""
Golf policy player with ROS2 joint-state mirroring to real myCobot280 Pi.

Full motion sequence when hole N is selected:
  1. Robot animates HOME → INIT pose
  2. env.reset() initialises the episode
  3. Policy runs until episode ends (ball-in-hole or timeout)
  4. Robot animates current pose → HOME
  5. Wait for next hole input

With --ros2: every joint write (animation + policy steps) is written to
/tmp/golf_joints.json.  A separate relay (golf_joint_relay.py, system Python
3.10) reads that file and publishes to ROS2.  This sidesteps the Python 3.11
(Isaac Lab) / Python 3.10 (ROS2 Humble) incompatibility.

Run inside Isaac Lab:
    python scripts/golf_play_ros2_bridge.py --task Mini-G-Golf-v0 --ros2

Hole selector (separate terminal, no Isaac Lab needed):
    python scripts/golf_hole_selector.py
"""

import argparse
import glob
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--task",           type=str,   default="Mini-G-Golf-v0")
parser.add_argument("--checkpoint",     type=str,   default=None)
parser.add_argument("--num_envs",       type=int,   default=1)
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--real_time",      action="store_true",
                    help="Pace policy to 25 Hz real-time.")
parser.add_argument("--transition_sec", type=float, default=1.5,
                    help="Duration of home↔init joint animation (seconds).")
parser.add_argument("--ros2",           action="store_true",
                    help="Write joint angles to /tmp/golf_joints.json for ROS2 relay.")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Joint file bridge ────────────────────────────────────────────────────────
# rclpy cannot be imported here: Isaac Lab uses Python 3.11 but ROS2 Humble
# only ships Python 3.10 bindings.  Instead, Isaac Sim writes joint angles to
# a JSON file; golf_joint_relay.py (system Python 3.10) reads and publishes.
JOINTS_FILE = "/tmp/golf_joints.json"

# ── Post-launch imports ───────────────────────────────────────────────────────
import gymnasium as gym
import skrl
import torch
from packaging import version

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(f"Need skrl>={SKRL_VERSION}, got {skrl.__version__}")
    sys.exit(1)

from skrl.utils.runner.torch import Runner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import Mini_G.tasks    # noqa: F401


# ==============================================================================
#  Geometry — must match mini_g_env_cfg.py
# ==============================================================================

BALL_POS_LOCAL = (0.22868, 0.0351, 0.021)
HOLE_DIST      = 0.28
HOLE_Z         = 0.002

HOLE_ANGLE_DEG = {1: 85.0, 2: 90.0, 3: 95.0}

_JOINT_NAMES = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
]

HOME_POSE_RAD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Speed for each phase — higher = faster motor, less lag
ANIM_SPEED   = 100   # home↔init interpolation (smooth but not too slow)
POLICY_SPEED = 100   # policy strokes (fast to track the sim swing)


def init_pose_rad(hole_id: int) -> list:
    """Policy init pose: arm behind ball, EE pointing toward hole."""
    phi = math.radians(HOLE_ANGLE_DEG[hole_id])
    return [
        math.radians(  0.0),
        math.radians(-100.0),
        math.radians( -38.8),
        math.radians(  0.0),
        phi,
        0.0,
    ]


def _hole_local(hole_id: int) -> tuple:
    phi = math.radians(HOLE_ANGLE_DEG[hole_id])
    return (
        BALL_POS_LOCAL[0] + HOLE_DIST * math.cos(phi),
        BALL_POS_LOCAL[1] + HOLE_DIST * math.sin(phi),
        HOLE_Z,
    )


# ==============================================================================
#  JSON command bridge
# ==============================================================================

HOLE_FILE = "/tmp/golf_hole.json"


def _write_file(hole_id: int) -> None:
    tmp = HOLE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"hole_id": hole_id}, f)
    os.replace(tmp, HOLE_FILE)


def poll_command() -> int:
    try:
        return int(json.load(open(HOLE_FILE)).get("hole_id", 0))
    except Exception:
        return 0


# ==============================================================================
#  Robot motion helpers
# ==============================================================================

def _get_joint_ids(env):
    robot = env.unwrapped.scene["robot"]
    ids, _ = robot.find_joints(_JOINT_NAMES)
    return ids


def publish_joints(env, speed: int = POLICY_SPEED) -> None:
    """Write current sim joint positions (degrees) + speed to the ROS2 relay file."""
    if not args_cli.ros2:
        return
    robot      = env.unwrapped.scene["robot"]
    jids       = _get_joint_ids(env)
    angles_deg = [math.degrees(a.item()) for a in robot.data.joint_pos[0, jids]]
    tmp = JOINTS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"angles_deg": angles_deg, "speed": speed}, f)
    os.replace(tmp, JOINTS_FILE)   # atomic on Linux


def animate_joints(env, target_rad: list, duration_s: float) -> None:
    """
    Smoothly interpolate joints from current to target_rad over duration_s.
    Publishes each frame to ROS2 so the real arm follows.
    """
    robot   = env.unwrapped.scene["robot"]
    device  = env.unwrapped.device
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    jids    = _get_joint_ids(env)

    current = robot.data.joint_pos[0, jids].clone()
    target  = torch.tensor(target_rad, device=device, dtype=torch.float32)

    steps  = max(1, int(duration_s * 25))
    step_t = duration_s / steps

    for i in range(steps):
        alpha  = (i + 1) / steps
        interp = (current + alpha * (target - current)).unsqueeze(0)
        zeros  = torch.zeros_like(interp)
        with torch.inference_mode():
            robot.write_joint_state_to_sim(interp, zeros,
                                           joint_ids=jids, env_ids=env_ids)
        simulation_app.update()
        publish_joints(env, speed=ANIM_SPEED)
        time.sleep(step_t)


def set_home(env) -> None:
    animate_joints(env, HOME_POSE_RAD, duration_s=args_cli.transition_sec)


def set_init_pose(env, hole_id: int) -> None:
    animate_joints(env, init_pose_rad(hole_id), duration_s=args_cli.transition_sec)


def set_hole_marker(env, hole_id: int) -> None:
    """Teleport the hole_marker rigid body to the selected position."""
    local     = _hole_local(hole_id)
    unwrapped = env.unwrapped
    device    = unwrapped.device
    origin    = unwrapped.scene.env_origins[0]

    pos  = torch.tensor([[local[0] + origin[0].item(),
                          local[1] + origin[1].item(),
                          local[2] + origin[2].item()]], device=device)
    rot  = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    pose = torch.cat([pos, rot], dim=-1)

    env.unwrapped.scene["hole_marker"].write_root_pose_to_sim(pose)


# ==============================================================================
#  Checkpoint finder
# ==============================================================================

_DEFAULT_CHECKPOINT = (
    "/home/goel/code-goel/MyCobot_RL/Mini_G/logs/skrl/mini_g_golf"
    "/2026-04-02_17-51-05_ppo_torch_ppo/checkpoints/best_agent.pt"
)


def find_latest_checkpoint() -> str:
    # Try the known-good checkpoint first
    if os.path.exists(_DEFAULT_CHECKPOINT):
        print(f"[INFO] Using default checkpoint: {_DEFAULT_CHECKPOINT}")
        return _DEFAULT_CHECKPOINT
    # Fall back to newest best_agent.pt under logs/
    search = os.path.expanduser("~/code-goel/MyCobot_RL/Mini_G/logs")
    files  = glob.glob(os.path.join(search, "**/best_agent.pt"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No best_agent.pt found under {search}")
    latest = max(files, key=os.path.getmtime)
    print(f"[INFO] Auto-selected checkpoint: {latest}")
    return latest


# ==============================================================================
#  Main
# ==============================================================================

@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, experiment_cfg: dict):

    # ── Build env & runner ───────────────────────────────────────────────────
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed           = args_cli.seed
    experiment_cfg["seed"] = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    experiment_cfg["trainer"]["close_environment_at_exit"]       = False
    experiment_cfg["agent"]["experiment"]["write_interval"]      = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    runner = Runner(env, experiment_cfg)

    # ── Load checkpoint ──────────────────────────────────────────────────────
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        log_root = os.path.join("logs", "skrl",
                                experiment_cfg["agent"]["experiment"]["directory"])
        try:
            resume_path = get_checkpoint_path(
                os.path.abspath(log_root),
                run_dir=".*_ppo_torch",
                other_dirs=["checkpoints"],
            )
        except Exception:
            resume_path = find_latest_checkpoint()

    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    # ── Boot ─────────────────────────────────────────────────────────────────
    _obs, _ = env.reset()
    set_home(env)
    _write_file(0)

    print("\n" + "="*60)
    print("  Golf Demo  — robot at home, waiting for hole selection")
    print("  Run in another terminal:")
    print("      python scripts/golf_hole_selector.py")
    print("  Then type  1 / 2 / 3  to start a stroke.")
    if args_cli.ros2:
        print(f"  ROS2: writing joints → {JOINTS_FILE}  (run golf_joint_relay.py to forward)")
    print("="*60 + "\n")

    # ── State machine ─────────────────────────────────────────────────────────
    WAITING = "waiting"
    RUNNING = "running"
    state      = WAITING
    episode    = 0
    step_count = ep_reward = 0
    obs        = _obs

    while simulation_app.is_running():

        # ── WAITING ───────────────────────────────────────────────────────────
        if state == WAITING:
            hole_id = poll_command()
            if hole_id in (1, 2, 3):
                _write_file(0)
                pos = _hole_local(hole_id)
                print(f"[Golf] Hole {hole_id} selected "
                      f"({pos[0]:.3f}, {pos[1]:.3f}) m")

                print("       Moving to init pose ...")
                set_init_pose(env, hole_id)

                print("       Starting stroke ...")
                obs, _ = env.reset()
                with torch.inference_mode():
                    set_hole_marker(env, hole_id)

                step_count = 0
                ep_reward  = 0.0
                state      = RUNNING
            else:
                simulation_app.update()
                time.sleep(0.05)

        # ── RUNNING ───────────────────────────────────────────────────────────
        elif state == RUNNING:
            loop_start = time.time()

            with torch.inference_mode():
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
                obs, rewards, terminated, truncated, _ = env.step(actions)

            publish_joints(env)

            step_count += 1
            ep_reward  += rewards[0].item()
            done        = (terminated | truncated)[0].item()

            if done:
                episode += 1
                success  = terminated[0].item()
                print(
                    f"[Episode {episode:3d}]  "
                    f"{'HOLE IN ONE! ' if success else 'timeout     '}  "
                    f"steps={step_count:3d}  reward={ep_reward:+7.2f}"
                )

                print("       Returning to home pose ...")
                set_home(env)
                publish_joints(env, speed=ANIM_SPEED)   # publish final home position

                print("[Golf] Robot at home. Enter 1 / 2 / 3 for next stroke.\n")
                state = WAITING

            elif args_cli.real_time:
                sleep_s = dt - (time.time() - loop_start)
                if sleep_s > 0:
                    time.sleep(sleep_s)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
