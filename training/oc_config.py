"""
Option-Critic Configuration for HPRL Training

This file mirrors the structure of *ddpg_config.py* and *sac_config.py* so that
all helper functions (`get_config`, `print_config`, `save_config`) work
unchanged.  Only OC-specific hyper-parameters live under the `'oc'` key.
"""

from __future__ import annotations
import os
from typing import Any, Dict

# -----------------------------------------------------------------------------
# Base configuration (values chosen to match common OC practice + HPRL defaults)
# -----------------------------------------------------------------------------

OC_CONFIG: Dict[str, Any] = {
    "training": {
        "max_episodes": 2000,
        "eval_frequency": 3,
        "save_frequency": 1000,
        "log_frequency": 1,
        "warmup_episodes": 50,            # random policy episodes before updates
        "max_episode_steps": 500,
    },

    # ------------------------  Option-Critic hyper-parameters  ----------------
    "oc": {
        "actor_lr": 3e-5,                 # θ (intra-option policy) learning rate
        "critic_lr": 3e-5,                # w (Q-critic) learning rate
        "beta_lr": 1e-4,                  # ϕ (termination) learning rate
        "tau": 0.01,                      # soft-update for target networks
        "gamma": 0.95,                    # discount
        "buffer_size": 50_000,
        "batch_size": 32,
        "macro_steps": 5,                 # |H| (programme count per episode)
        "max_program_steps": 50,
        "num_options": 4,                 # number of options
        "entropy_coef": 0.01,             # entropy bonus for π_θ
        "termination_reg": 0.01,          # β regulariser (encourage persistence)
        "gradient_clip": 10.0,
    },

    # --------------------------  VAE integration params  ----------------------
    "vae": {
        "latent_scaling": 1.5,
        "use_latent_clipping": True,
        "latent_clip_range": [-4.0, 4.0],
        "generation_temperature": 1.0,
    },

    # ---------------------------  Reward shaping  -----------------------------
    "rewards": {
        'success_bonus': 1.0,       # Bonus for successful program execution
        'step_penalty': -0.001,     # Small penalty per execution step
        'failure_penalty': -0.1,    # Penalty for failed execution (was -0.5)
        'timeout_penalty': -0.01,   # Penalty for program timeout
        'invalid_program_penalty': -1,  # Penalty for invalid programs
        'executable_bonus': 0.2,    # NEW: Bonus for syntactically correct/executable programs
        'efficiency_reward': 0.1,   # Reward for shorter successful programs
        'progress_reward_scale': 1.0,  # Scale factor for intermediate progress
    },

    # ---------------------------  Network sizes  ------------------------------
    "networks": {
        "actor": {
            "hidden_dims": [512, 256, 128],
            "dropout": 0.1,
            "activation": "relu",
            "output_activation": "tanh",
        },
        "critic": {
            "hidden_dims": [512, 256, 128],
            "dropout": 0.1,
            "activation": "relu",
        },
        "termination": {
            "hidden_dims": [256, 128],     # for β_ϕ(s) heads
            "dropout": 0.1,
            "activation": "relu",
        },
        "conv_channels": [32, 64, 32],
    },

    # ------------------------  Environment tweaks  ---------------------------
    "environment": {
        "grid_size": (8, 8),
        "timeout_steps": 100,
        "reset_on_success": True,
        "reward_normalization": False,
    },

    # -----------------------------  Debug flags  ------------------------------
    "debug": {
        'log_program_samples': True,    # Log decoded programs
        'log_latent_stats': True,       # Log latent embedding statistics
        'save_execution_traces': False, # Save detailed execution traces
        'verbose_evaluation': True,     # Detailed evaluation logging
        'plot_frequency': 100,          # How often to generate plots
    },
}

# -----------------------------------------------------------------------------
# Task-specific overrides (inherit from DDPG defaults so OC keeps same tasks)
# -----------------------------------------------------------------------------

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "harvester": {
        "rewards": {"success_bonus": 2.0, "efficiency_reward": 0.2},
        "oc": {"macro_steps": 5, "max_program_steps": 60},
    },
    "cleanHouse": {
        "rewards": {"success_bonus": 3.0, "step_penalty": -0.002},
        "oc": {"macro_steps": 7, "max_program_steps": 80},
    },
    "fourCorners": {
        "rewards": {"success_bonus": 2.5, "efficiency_reward": 0.3},
        "oc": {"macro_steps": 4, "max_program_steps": 40},
    },
    "randomMaze": {
        "rewards": {"success_bonus": 4.0},
        "oc": {"macro_steps": 8, "max_program_steps": 100},
    },
}

# -----------------------------------------------------------------------------
# Helper utilities (copied from ddpg_config) so the external API is identical
# -----------------------------------------------------------------------------

def deep_merge(base: Dict, override: Dict) -> Dict:
    res = base.copy()
    for k, v in override.items():
        if k in res and isinstance(res[k], dict) and isinstance(v, dict):
            res[k] = deep_merge(res[k], v)
        else:
            res[k] = v
    return res


def validate(cfg: Dict[str, Any]):
    for sec in ("training", "oc", "rewards", "networks"):
        if sec not in cfg:
            raise ValueError(f"Missing config section: {sec}")
    if cfg["oc"]["actor_lr"] <= 0 or cfg["oc"]["critic_lr"] <= 0:
        raise ValueError("Learning rates must be positive")
    if cfg["rewards"]["failure_penalty"] > 0:
        raise ValueError("Failure penalty must be negative")
    if not cfg["networks"]["actor"]["hidden_dims"]:
        raise ValueError("Actor network needs hidden layers")


def get_config(task: str = "harvester", custom: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = deep_merge({}, OC_CONFIG)
    if task in TASK_CONFIGS:
        cfg = deep_merge(cfg, TASK_CONFIGS[task])
    if custom:
        cfg = deep_merge(cfg, custom)
    validate(cfg)
    return cfg


def print_config(cfg: Dict[str, Any]):
    """Pretty print utility used by trainers"""
    import pprint
    pprint.pprint(cfg, width=110, compact=True)


def save_config(cfg: Dict[str, Any], filepath: str):
    import json
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"OC config saved to {filepath}")



# -----------------------------------------------------------------------------
# Quick self-test when run directly
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = get_config("harvester")
    print_config(cfg)
    save_config(cfg, "./oc_config_test.json")
    print("✅ OC configuration module sanity-checked.")