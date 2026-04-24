# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class GolfPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for the myCobot280 golf task.

    Network architecture
    --------------------
    Actor  : MLP 256-128-64  (tanh activations)
    Critic : MLP 256-128-64  (tanh activations)
    Obs dim: 27  –  Action dim: 6

    Training schedule
    -----------------
    ~2 000 iterations × 2048 envs × 200 steps ≈ 800 M environment interactions
    on a single A100/RTX4090 this takes roughly 3–6 hours.
    """

    num_steps_per_env: int = 24       # rollout horizon per env
    max_iterations: int = 2000        # total PPO update cycles

    save_interval: int = 100          # checkpoint every N iterations
    experiment_name: str = "mini_g_golf"
    empirical_normalization: bool = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",      # LR schedule
        gamma=0.99,               # discount factor
        lam=0.95,                 # GAE lambda
        desired_kl=0.01,
        max_grad_norm=1.0,
    )