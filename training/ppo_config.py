"""
PPO Configuration for HPRL Training
===================================

This file mirrors the structure of *ddpg_config.py* so that the trainer
and CLI utilities work interchangeably.  All hyper-parameters match the
official HPRL repository’s PPO meta-policy experiments.
"""

import os
from typing import Dict, Any

# ──────────────────────────────────────────────────────────────
#   Base configuration
# ──────────────────────────────────────────────────────────────
PPO_CONFIG: Dict[str, Any] = {
    # Training parameters
    "training": {
        "max_episodes": 2000,
        "eval_frequency": 100,
        "save_frequency": 1000,
        "log_frequency": 10,
        "max_episode_steps": 5,      # safety-cap (not usually hit)
    },

    # PPO-specific hyper-parameters
    "ppo": {
            # --- optimiser ---------------------------------------------------
            "lr"            : 7e-4,          # ←  HPRL default
            "eps"           : 1e-5,          #   (Adam ε, exposed here for completeness)
            # --- RL coefficients ---------------------------------------------
            "gamma"         : 0.99,
            "gae_lambda"    : 0.95,
            "clip_eps"      : 0.20,          # PPO clip-ratio
            "entropy_coef"  : 0.01,
            "value_coef"    : 0.5,
            "max_grad_norm" : 0.5,
            # --- roll-out & SGD schedule -------------------------------------
            # 16 parallel env-processes × 50 steps  = 800 transitions / update
            "rollout_steps" : 800,
            "ppo_epochs"    : 3,
            "minibatch_size": 5,
            # --- hierarchical meta-policy ------------------------------------
            "macro_steps"        : 5,        # |H| in the paper
            "max_program_steps"  : 50,
    },

    # VAE integration (identical to DDPG)
    "vae": {
        "latent_scaling":       1.5,
        "use_latent_clipping":  True,
        "latent_clip_range":    [-3.0, 3.0],
        "generation_temperature": 1.0,
    },

    # Reward shaping (same defaults)
    "rewards": {
        "success_bonus":          1.0,
        "step_penalty":          -0.001,
        "failure_penalty":       -0.1,
        "timeout_penalty":       -0.05,
        "invalid_program_penalty": -0.2,
        "efficiency_reward":      0.1,
        "progress_reward_scale":  0.5,
    },

    # Network architecture
    "networks": {
        "actor": {
            "hidden_dims": [512, 256, 128],
            "dropout":     0.1,
            "activation":  "relu",
            "output_activation": "tanh",
        },
        "critic": {
            "hidden_dims": [512, 256, 128],
            "dropout":     0.1,
            "activation":  "relu",
        },
        "conv_channels": [32, 64, 32],
    },

    # Environment
    "environment": {
        "grid_size":          (8, 8),
        "timeout_steps":      100,
        "reset_on_success":   True,
        "reward_normalization": False,
    },

    # Debug / logging
    "debug": {
        "log_program_samples":  True,
        "log_latent_stats":     True,
        "save_execution_traces": False,
        "verbose_evaluation":   True,
        "plot_frequency":       100,
    },
}

# ──────────────────────────────────────────────────────────────
#   Task-specific overrides
# ──────────────────────────────────────────────────────────────
TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "harvester": {
        "rewards": {"success_bonus": 2.0, "efficiency_reward": 0.2},
        "ppo":     {"macro_steps": 5, "max_program_steps": 60},
    },
    "cleanHouse": {
        "rewards": {"success_bonus": 3.0, "step_penalty": -0.002},
        "ppo":     {"macro_steps": 7, "max_program_steps": 80},
    },
    "fourCorners": {
        "rewards": {"success_bonus": 2.5, "efficiency_reward": 0.3},
        "ppo":     {"macro_steps": 4, "max_program_steps": 40},
    },
    "randomMaze": {
        "rewards": {"success_bonus": 4.0, "progress_reward_scale": 1.0},
        "ppo":     {"macro_steps": 8, "max_program_steps": 100},
    },
    "stairClimber": {
        "rewards": {"success_bonus": 1.5, "efficiency_reward": 0.4},
        "ppo":     {"macro_steps": 3, "max_program_steps": 30},
    },
    "topOff": {
        "rewards": {"success_bonus": 2.0, "step_penalty": -0.001},
        "ppo":     {"macro_steps": 6, "max_program_steps": 70},
    },
}

# Curriculum schedule (kept for symmetry; rarely used with PPO)
CURRICULUM_CONFIG = {
    "enabled": False,
    "stages": [
        {
            "episodes": [0, 200],
            "config_overrides": {"ppo.rollout_steps": 1024},
        },
        {
            "episodes": [200, 500],
            "config_overrides": {"ppo.rollout_steps": 2048},
        },
    ],
}

# ──────────────────────────────────────────────────────────────
#   Helper functions  (mostly identical to ddpg_config.py)
# ──────────────────────────────────────────────────────────────
def deep_merge(base: Dict, override: Dict) -> Dict:
    out = base.copy()
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if k in out and isinstance(out[k], dict) and isinstance(v, dict) else v
    return out

def validate(cfg: Dict[str, Any]):
    for sec in ("training", "ppo", "rewards", "networks"):
        if sec not in cfg:
            raise ValueError(f"Missing section '{sec}'")
    if cfg["ppo"]["lr"] <= 0:
        raise ValueError("Learning rate must be positive")
    if cfg["rewards"]["failure_penalty"] > 0:
        raise ValueError("failure_penalty should be negative")
    if not cfg["networks"]["actor"]["hidden_dims"]:
        raise ValueError("Actor hidden_dims empty")
    print("✅ PPO config validation passed")

def get_config(task: str = "harvester", custom: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = PPO_CONFIG.copy()
    if task in TASK_CONFIGS:
        cfg = deep_merge(cfg, TASK_CONFIGS[task])
    if custom:
        cfg = deep_merge(cfg, custom)
    validate(cfg)
    return cfg

def print_config(cfg: Dict[str, Any]):
    import pprint, textwrap
    print("=" * 60)
    print("PPO CONFIGURATION")
    print("=" * 60)
    pprint.pprint(cfg, width=120, compact=False)
    print("=" * 60)

def save_config(cfg: Dict[str, Any], path: str):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Configuration saved to {path}")

# ──────────────────────────────────────────────────────────────
#   Quick test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = get_config("harvester")
    print_config(cfg)