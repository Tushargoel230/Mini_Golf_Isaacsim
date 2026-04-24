# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# Task registration for the myCobot280 Golf Push environment.
# ==============================================================================

import gymnasium as gym

from . import agents
from .mini_g_env_cfg import MiniGolfEnvCfg, MiniGolfEnvCfg_Visual, MiniGolfEnvCfg_PLAY

_KWARGS = {
    "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GolfPPORunnerCfg",
    "skrl_cfg_entry_point":   f"{agents.__name__}:skrl_ppo_cfg.yaml",
    "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
}

# ── Fast headless training (no camera, no --enable_cameras needed) ────────────
gym.register(
    id="Mini-G-Golf-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": MiniGolfEnvCfg, **_KWARGS},
)

# ── GUI with top-down camera viewport (requires --enable_cameras) ─────────────
gym.register(
    id="Mini-G-Golf-Visual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": MiniGolfEnvCfg_Visual, **_KWARGS},
)

# ── Evaluation (no noise, longer episodes) ────────────────────────────────────
gym.register(
    id="Mini-G-Golf-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": MiniGolfEnvCfg_PLAY, **_KWARGS},
)