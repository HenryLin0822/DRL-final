
"""SAC Configuration for HPRL Training

Mirrors `ddpg_config.py` so the same helper functions (`get_config`,
`print_config`, `save_config`) work unchanged.  All SAC‑specific
hyper‑parameters live under the `'sac'` key.
"""

import os
from typing import Dict, Any

# Base configuration (copied & tweaked from DDPG)
# Auto-generated SAC config matching HPRL paper
SAC_CONFIG: Dict[str, Any] = {
    # Training parameters
    'training': {
        'max_episodes': 2000,
        'eval_frequency': 100,
        'save_frequency': 1000,
        'log_frequency': 10,
        'warmup_episodes': 100,
        'max_episode_steps': 500,
    },

    # SAC-specific agent hyper-parameters
    'sac': {
        'actor_lr': 0.001,
        'critic_lr': 0.001,
        'alpha_lr': 0.001,
        'tau': 0.005,
        'gamma': 0.99,
        'buffer_size': 100000,
        'batch_size': 256,
        'init_temperature': 0.2,
        'target_entropy_scale': 1.0,
        'macro_steps': 5,
        'max_program_steps': 50,
        'update_frequency': 1,
        'gradient_clip': 10.0,
        'latent_embedding_size': 64,
        'latent_loss_coefficient': 0.1,
    },

    # VAE integration parameters
    'vae': {
        'latent_scaling': 1.5,
        'use_latent_clipping': True,
        'latent_clip_range': [-3.0, 3.0],
        'generation_temperature': 1.0,
    },

    # Reward shaping parameters
    'rewards': {
        'success_bonus': 1.0,
        'step_penalty': -0.001,
        'failure_penalty': -0.1,
        'timeout_penalty': -0.05,
        'invalid_program_penalty': -0.2,
        'efficiency_reward': 0.1,
        'progress_reward_scale': 0.5,
    },

    # Network architecture
    'networks': {
        'actor': {
            'hidden_dims': [256],
            'dropout': 0.0,
            'activation': 'tanh',
            'output_activation': 'tanh',
            'gru_layers': 1,
        },
        'critic': {
            'hidden_dims': [256],
            'dropout': 0.0,
            'activation': 'tanh',
            'gru_layers': 1,
        },
        'conv_channels': [32, 64, 32],
    },

    # Environment settings
    'environment': {
        'grid_size': (8, 8),
        'timeout_steps': 100,
        'reset_on_success': True,
        'reward_normalization': False,
    },

    # Debugging / monitoring
    'debug': {
        'log_program_samples': True,
        'log_latent_stats': True,
        'save_execution_traces': False,
        'verbose_evaluation': True,
        'plot_frequency': 100,
    },

    # Exploration strategies
    'exploration': {
        'strategy': 'ou_noise',
        'ou_theta': 0.15,
        'ou_sigma': 0.2,
        'ou_dt': 1e-2,
        'gaussian_std': 0.1,
        'epsilon_start': 1.0,
        'epsilon_end': 0.1,
        'epsilon_decay': 0.995,
    }
}

# Task-specific configurations
TASK_CONFIGS = {
    'harvester': {
        'rewards': {
            'success_bonus': 2.0,
            'efficiency_reward': 0.2,
        },
        'sac': {
            'macro_steps': 5,
            'max_program_steps': 60,
        }
    },
    
    'cleanHouse': {
        'rewards': {
            'success_bonus': 3.0,
            'step_penalty': -0.002,
        },
        'sac': {
            'macro_steps': 7,
            'max_program_steps': 80,
        }
    },
    
    'fourCorners': {
        'rewards': {
            'success_bonus': 2.5,
            'efficiency_reward': 0.3,
        },
        'sac': {
            'macro_steps': 4,
            'max_program_steps': 40,
        }
    },
    
    'randomMaze': {
        'rewards': {
            'success_bonus': 4.0,
            'progress_reward_scale': 1.0,
        },
        'sac': {
            'macro_steps': 8,
            'max_program_steps': 100,
        }
    },
    
    'stairClimber': {
        'rewards': {
            'success_bonus': 1.5,
            'efficiency_reward': 0.4,
        },
        'sac': {
            'macro_steps': 3,
            'max_program_steps': 30,
        }
    },
    
    'topOff': {
        'rewards': {
            'success_bonus': 2.0,
            'step_penalty': -0.001,
        },
        'sac': {
            'macro_steps': 6,
            'max_program_steps': 70,
        }
    }
}

# Curriculum learning schedule (optional)
CURRICULUM_CONFIG = {
    'enabled': False,
    'stages': [
        {
            'episodes': [0, 200],
            'config_overrides': {
                'sac.noise_std': 0.5,
                'rewards.failure_penalty': -0.05,
                'sac.max_program_steps': 30,
            }
        },
        {
            'episodes': [200, 500],
            'config_overrides': {
                'sac.noise_std': 0.3,
                'rewards.failure_penalty': -0.1,
                'sac.max_program_steps': 50,
            }
        },
        {
            'episodes': [500, float('inf')],
            'config_overrides': {
                'sac.noise_std': 0.2,
                'rewards.failure_penalty': -0.2,
                'sac.max_program_steps': 100,
            }
        }
    ]
}


# --- helper functions (reuse from ddpg_config) ------------------
def deep_merge_dicts(base: Dict, override: Dict) -> Dict:
    """Recursively merge dictionaries (same helper as DDPG)."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def validate_config(cfg: Dict[str, Any]) -> None:
    required = ['training', 'sac', 'rewards', 'networks']
    for s in required:
        if s not in cfg:
            raise ValueError(f"Missing config section: {s}")
    if cfg['sac']['actor_lr'] <= 0 or cfg['sac']['critic_lr'] <= 0:
        raise ValueError('Learning rates must be positive')
    if cfg['rewards']['failure_penalty'] > 0:
        raise ValueError('Failure penalty must be negative')
    if not cfg['networks']['actor']['hidden_dims']:
        raise ValueError('Actor network needs hidden layers')
    if not cfg['networks']['critic']['hidden_dims']:
        raise ValueError('Critic network needs hidden layers')


def get_config(task: str = 'harvester', custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Return config (task‑agnostic for now, but hook provided)."""
    config = SAC_CONFIG.copy()
    if task in TASK_CONFIGS:
        task_config = TASK_CONFIGS[task]
        config = deep_merge_dicts(config, task_config)

    if custom_config:
        config = deep_merge_dicts(config, custom_config)

    validate_config(config)
    return config


def print_config(config: Dict[str, Any]) -> None:
    """Pretty print configuration"""
    print("=" * 60)
    print("SAC CONFIGURATION")
    print("=" * 60)
    
    def print_section(section_name: str, section_data: Dict, indent: int = 0):
        prefix = "  " * indent
        print(f"{prefix}{section_name.upper()}:")
        
        for key, value in section_data.items():
            if isinstance(value, dict):
                print_section(key, value, indent + 1)
            else:
                print(f"{prefix}  {key}: {value}")
        print()
    
    for section_name, section_data in config.items():
        if isinstance(section_data, dict):
            print_section(section_name, section_data)
        else:
            print(f"{section_name}: {section_data}")
    
    print("=" * 60)


def save_config(cfg: Dict[str, Any], filepath: str):
    import json, os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'SAC config saved to {filepath}')
