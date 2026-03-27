# Copyright (c) 2025, Tushar Goel, TU Dortmund.
# SKRL PPO configuration – Mini-G Golf Hole-In-One
# Action space : 3  (Δx, Δy, Δz EE position deltas via HIGH_PD IK)
# Observation  : 30 (joint_pos×6 + joint_vel×6 + ee_pos×3 + ball_pos×3
#                    + ball_vel×3 + ee→ball×3 + ball→hole×3)
#
# Actuator config: MYCOBOT_280_HIGH_PD_CFG (stiffness=800, damping=50)
# IK scale: 0.02 m/step  at 25Hz = 0.5 m/s max EE speed
# ==============================================================================

seed: 42

# ── Models ────────────────────────────────────────────────────────────────────
models:
  separate: false
  policy:
    class: GaussianMixin
    clip_actions: false
    clip_log_std: true
    min_log_std: -20.0
    max_log_std:  2.0
    initial_log_std: 0.0
    network:
      - name: net
        input: STATES
        layers: [256, 128, 64]
        activations: elu
    output: ACTIONS

  value:
    class: DeterministicMixin
    clip_actions: false
    network:
      - name: net
        input: STATES
        layers: [256, 128, 64]
        activations: elu
    output: ONE

# ── PPO Agent ─────────────────────────────────────────────────────────────────
agent:
  class: PPO          # required by skrl Runner
  rollouts: 24
  learning_epochs: 5
  mini_batches: 4
  discount_factor: 0.99
  lambda: 0.95
  learning_rate: 3.0e-4      # lower LR – IK actions are more structured
  learning_rate_scheduler: KLAdaptiveLR
  learning_rate_scheduler_kwargs:
    kl_threshold: 0.008
  random_timesteps: 0
  learning_starts: 0
  grad_norm_clip: 1.0
  ratio_clip: 0.2
  value_clip: 0.2
  clip_predicted_values: true
  entropy_coef: 0.01         # slightly higher entropy to encourage EE exploration
  entropy_loss_scale: 0.01
  value_loss_scale: 1.0
  kl_threshold: 0.0
  # Observation normalisation (dim=30: +3 ee_to_approach_target added)
  state_preprocessor: RunningStandardScaler
  state_preprocessor_kwargs:
    size: 30
  value_preprocessor: RunningStandardScaler
  value_preprocessor_kwargs:
    size: 1
  # ── Experiment / logging ──────────────────────────────────────────────────
  experiment:
    directory: "mini_g_golf"
    experiment_name: "ppo"
    write_interval: 500
    checkpoint_interval: 5000

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer:
  timesteps: 24000000    # 24 M steps – IK converges ~2× faster than joint-space
  environment_info: log