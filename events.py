# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# rl_games PPO configuration – Mini-G Golf Hole-In-One
# ==============================================================================

params:
  seed: 42

  algo:
    name: a2c_continuous

  model:
    name: continuous_a2c_logstd

  network:
    name: actor_critic
    separate: false
    space:
      continuous:
        mu_activation: None
        sigma_activation: None
        mu_init:
          name: default
        sigma_init:
          name: const_initializer
          val: 0
        fixed_sigma: true

    mlp:
      units: [256, 128, 64]
      activation: elu
      initializer:
        name: default
      regularizer:
        name: None

  load_checkpoint: false
  load_path: ''

  config:
    name: mini_g_golf
    full_experiment_name: mini_g_golf_ppo
    env_name: rlgpu
    device: cuda:0
    device_name: cuda:0
    multi_gpu: false
    ppo: true
    mixed_precision: false
    normalize_input: true
    normalize_value: true
    value_bootstrap: true
    num_actors: -1          # set at runtime from num_envs
    reward_shaper:
      scale_value: 1.0
    normalize_advantage: true
    gamma: 0.99
    tau: 0.95
    learning_rate: 1.0e-3
    lr_schedule: adaptive
    schedule_type: standard
    kl_threshold: 0.008
    score_to_win: 200
    max_epochs: 2000
    save_best_after: 100
    save_frequency: 100
    print_stats: true
    entropy_coef: 0.005
    truncate_grads: true
    grad_norm: 1.0
    e_clip: 0.2
    horizon_length: 24
    minibatch_size: 12288
    mini_epochs: 5
    critic_coef: 1.0
    clip_value: true
    seq_length: 4
    bounds_loss_coef: 0.0001