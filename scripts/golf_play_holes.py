#!/usr/bin/env python3
# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# SPDX-License-Identifier: BSD-3-Clause
"""
Golf policy player with manual hole selection and home-position pausing.

Full motion sequence when hole N is selected:
  1. Robot animates from HOME pose → INIT pose (policy start configuration)
  2. env.reset() initialises the episode properly
  3. Policy runs until episode ends (ball-in-hole or timeout)
  4. Robot animates from current pose → HOME pose
  5. Wait for next hole input

HOME pose : all joints 0° (robot standing upright)
INIT pose : [0°, -100°, -38.8°, 0°, φ, 0°] — arm behind ball, EE facing hole

Run inside Isaac Lab:
    python scripts/golf_play_holes.py --task Mini-G-Golf-v0 [--checkpoint PATH]

Run in a separate plain terminal:
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
parser.add_argument("--task",           type=str,  default="Mini-G-Golf-v0")
parser.add_argument("--checkpoint",     type=str,  default=None)
parser.add_argument("--num_envs",       type=int,  default=1)
parser.add_argument("--seed",           type=int,  default=42)
parser.add_argument("--real_time",      action="store_true",
                    help="Pace policy execution to 25 Hz real-time.")
parser.add_argument("--transition_sec", type=float, default=1.5,
                    help="Duration of home↔init joint animation (seconds).")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
#  Geometry
# ==============================================================================

BALL_POS_LOCAL = (0.22868, 0.0351, 0.021)
HOLE_DIST      = 0.28
HOLE_Z         = 0.002

HOLE_ANGLE_DEG = {1: 85.0, 2: 90.0, 3: 95.0}

# Joint names in the order they appear in the articulation
_JOINT_NAMES = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
]

# Home pose: all joints 0° — robot standing upright, away from table
HOME_POSE_RAD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def init_pose_rad(hole_id: int) -> list:
    """
    Policy init pose: arm behind ball, EE flat face pointing toward hole.
    Matches reset_arm_for_swing() in events.py.
    """
    phi = math.radians(HOLE_ANGLE_DEG[hole_id])
    return [
        math.radians(  0.0),   # joint2: arm in +X direction
        math.radians(-100.0),  # joint3
        math.radians( -38.8),  # joint4
        math.radians(  0.0),   # joint5
        phi,                   # joint6: EE face toward hole
        0.0,                   # joint6output
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
    """Return hole_id (1/2/3) if pending, else 0."""
    try:
        return int(json.load(open(HOLE_FILE)).get("hole_id", 0))
    except Exception:
        return 0


def consume_command() -> int:
    """Read and immediately clear the command file. Returns 0 if none."""
    val = poll_command()
    _write_file(0)
    return val


# ==============================================================================
#  Robot motion helpers
# ==============================================================================

def _get_joint_ids(env):
    robot = env.unwrapped.scene["robot"]
    ids, _ = robot.find_joints(_JOINT_NAMES)
    return ids


def animate_joints(env, target_rad: list, duration_s: float) -> None:
    """
    Smoothly animate the robot joints from their current position to target_rad.
    Advances the Isaac Sim frame at each step so the motion is visible.
    Runs OUTSIDE policy / env.step() — just raw joint writes + sim updates.
    """
    robot    = env.unwrapped.scene["robot"]
    device   = env.unwrapped.device
    env_ids  = torch.tensor([0], device=device, dtype=torch.long)
    jids     = _get_joint_ids(env)

    # Read current joint positions for the joints we care about
    current = robot.data.joint_pos[0, jids].clone()   # (6,)
    target  = torch.tensor(target_rad, device=device, dtype=torch.float32)

    steps = max(1, int(duration_s * 25))   # target 25 Hz
    step_t = duration_s / steps

    for i in range(steps):
        alpha  = (i + 1) / steps
        interp = (current + alpha * (target - current)).unsqueeze(0)  # (1,6)
        zeros  = torch.zeros_like(interp)
        with torch.inference_mode():
            robot.write_joint_state_to_sim(interp, zeros,
                                           joint_ids=jids, env_ids=env_ids)
        simulation_app.update()
        time.sleep(step_t)


def set_home(env) -> None:
    """Animate robot back to home (all-zero) pose."""
    animate_joints(env, HOME_POSE_RAD, duration_s=args_cli.transition_sec)


def set_init_pose(env, hole_id: int) -> None:
    """Animate robot from home to the policy init pose for the given hole."""
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

def find_latest_checkpoint() -> str:
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

    # ── Boot: do one env reset (builds scene), move to home, clear file ──────
    _obs, _ = env.reset()
    set_home(env)        # animate to home so robot is visible at start
    _write_file(0)       # clear any stale command from a previous run

    print("\n" + "="*60)
    print("  Golf Demo  — robot at home, waiting for hole selection")
    print("  Run in another terminal:")
    print("      python scripts/golf_hole_selector.py")
    print("  Then type  1 / 2 / 3  to start a stroke.")
    print("="*60 + "\n")

    # ── State machine ─────────────────────────────────────────────────────────
    WAITING = "waiting"
    RUNNING = "running"
    state   = WAITING
    episode = 0
    step_count = ep_reward = 0
    obs = _obs   # will be overwritten after each real reset

    while simulation_app.is_running():

        # ──────────────────────────────────────────────────────────────────────
        if state == WAITING:
            hole_id = poll_command()
            if hole_id in (1, 2, 3):
                _write_file(0)   # consume — must re-select for next stroke
                pos = _hole_local(hole_id)
                print(f"[Golf] Hole {hole_id} selected "
                      f"({pos[0]:.3f}, {pos[1]:.3f}) m")

                # Step 1: animate home → init pose (visible pre-stroke motion)
                print(f"       Moving to init pose ...")
                set_init_pose(env, hole_id)

                # Step 2: proper env reset (sets arm via reset_arm_for_swing,
                #         randomises hole, etc.) — we then override the hole
                print(f"       Starting stroke ...")
                obs, _ = env.reset()
                with torch.inference_mode():
                    set_hole_marker(env, hole_id)

                step_count = 0
                ep_reward  = 0.0
                state      = RUNNING
            else:
                # Idle: keep sim window responsive, no physics advancement
                simulation_app.update()
                time.sleep(0.05)

        # ──────────────────────────────────────────────────────────────────────
        elif state == RUNNING:
            loop_start = time.time()

            with torch.inference_mode():
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
                obs, rewards, terminated, truncated, _ = env.step(actions)

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

                # Step 3: animate current pose → home
                print("       Returning to home pose ...")
                set_home(env)

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
