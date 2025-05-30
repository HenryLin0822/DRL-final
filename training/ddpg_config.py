"""
DDPG Configuration for HPRL Training

This file contains all hyperparameters and configuration settings for training
the DDPG meta-policy in the HPRL framework.
"""

import os
from typing import Dict, Any

# Base configuration
DDPG_CONFIG = {
    # Training parameters
    'training': {
        'max_episodes': 2000,
        'eval_frequency': 100,
        'save_frequency': 1000,
        'log_frequency': 10,
        'warmup_episodes': 50,  # Episodes before starting updates
        'max_episode_steps': 500,  # Max steps per episode (safety)
    },
    
    # DDPG Agent parameters
    'ddpg': {
        'actor_lr': 3e-4,           # Higher learning rate for faster initial learning
        'critic_lr': 3e-4,          # Match actor lr
        'tau': 0.01,                # Faster target network updates
        'gamma': 0.95,              # Slightly lower discount for shorter horizons
        'buffer_size': 50000,       # Smaller buffer for faster cycling
        'batch_size': 32,           # Smaller batch for more frequent updates
        'noise_std': 0.3,           # Higher initial noise for exploration
        'noise_decay': 0.999,        # Slower noise decay
        'noise_min': 0.1,          # Minimum noise to maintain exploration
        'macro_steps': 5,           # |H| = 5 from paper
        'max_program_steps': 50,    # Shorter programs initially
        'update_frequency': 1,      # Update every step
        'gradient_clip': 10.0,      # Gradient clipping for stability
    },
    
    # VAE integration parameters
    'vae': {
        'latent_scaling': 1.5,      # Scale factor for latent embeddings
        'use_latent_clipping': True,
        'latent_clip_range': [-3.0, 3.0],  # Wider range for VAE latents
        'generation_temperature': 1.0,     # Temperature for program generation
    },
    
    # Reward shaping parameters
    'rewards': {
        'success_bonus': 1.0,       # Bonus for successful program execution
        'step_penalty': -0.001,     # Small penalty per execution step
        'failure_penalty': -0.1,    # Penalty for failed execution (was -0.5)
        'timeout_penalty': -0.05,   # Penalty for program timeout
        'invalid_program_penalty': -0.2,  # Penalty for invalid programs
        'efficiency_reward': 0.1,   # Reward for shorter successful programs
        'progress_reward_scale': 0.5,  # Scale factor for intermediate progress
    },
    
    # Network architecture
    'networks': {
        'actor': {
            'hidden_dims': [512, 256, 128],  # Larger networks
            'dropout': 0.1,
            'activation': 'relu',
            'output_activation': 'tanh',
        },
        'critic': {
            'hidden_dims': [512, 256, 128],  # Larger networks
            'dropout': 0.1,
            'activation': 'relu',
        },
        'conv_channels': [32, 64, 32],  # CNN architecture
    },
    
    # Environment settings
    'environment': {
        'grid_size': (8, 8),
        'timeout_steps': 100,
        'reset_on_success': True,
        'reward_normalization': False,
    },
    
    # Debugging and monitoring
    'debug': {
        'log_program_samples': True,    # Log decoded programs
        'log_latent_stats': True,       # Log latent embedding statistics
        'save_execution_traces': False, # Save detailed execution traces
        'verbose_evaluation': True,     # Detailed evaluation logging
        'plot_frequency': 100,          # How often to generate plots
    },
    
    # Exploration strategies
    'exploration': {
        'strategy': 'ou_noise',  # 'ou_noise', 'gaussian', 'epsilon_greedy'
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
        'ddpg': {
            'macro_steps': 5,
            'max_program_steps': 60,
        }
    },
    
    'cleanHouse': {
        'rewards': {
            'success_bonus': 3.0,
            'step_penalty': -0.002,
        },
        'ddpg': {
            'macro_steps': 7,  # More complex task
            'max_program_steps': 80,
        }
    },
    
    'fourCorners': {
        'rewards': {
            'success_bonus': 2.5,
            'efficiency_reward': 0.3,
        },
        'ddpg': {
            'macro_steps': 4,  # Can be solved in fewer steps
            'max_program_steps': 40,
        }
    },
    
    'randomMaze': {
        'rewards': {
            'success_bonus': 4.0,  # Harder task
            'progress_reward_scale': 1.0,
        },
        'ddpg': {
            'macro_steps': 8,
            'max_program_steps': 100,
            'noise_std': 0.4,  # More exploration needed
        }
    },
    
    'stairClimber': {
        'rewards': {
            'success_bonus': 1.5,
            'efficiency_reward': 0.4,
        },
        'ddpg': {
            'macro_steps': 3,  # Simple task
            'max_program_steps': 30,
        }
    },
    
    'topOff': {
        'rewards': {
            'success_bonus': 2.0,
            'step_penalty': -0.001,
        },
        'ddpg': {
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
                'ddpg.noise_std': 0.5,
                'rewards.failure_penalty': -0.05,  # Gentler penalties initially
                'ddpg.max_program_steps': 30,      # Shorter programs
            }
        },
        {
            'episodes': [200, 500],
            'config_overrides': {
                'ddpg.noise_std': 0.3,
                'rewards.failure_penalty': -0.1,
                'ddpg.max_program_steps': 50,
            }
        },
        {
            'episodes': [500, float('inf')],
            'config_overrides': {
                'ddpg.noise_std': 0.2,
                'rewards.failure_penalty': -0.2,
                'ddpg.max_program_steps': 100,
            }
        }
    ]
}


def get_config(task: str = 'harvester', custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get configuration for a specific task with optional overrides
    
    Args:
        task: Karel task name
        custom_config: Dictionary of custom configuration overrides
        
    Returns:
        Complete configuration dictionary
    """
    # Start with base config
    config = DDPG_CONFIG.copy()
    
    # Apply task-specific config
    if task in TASK_CONFIGS:
        task_config = TASK_CONFIGS[task]
        config = deep_merge_dicts(config, task_config)
    
    # Apply custom overrides
    if custom_config:
        config = deep_merge_dicts(config, custom_config)
    
    # Validate configuration
    validate_config(config)
    
    return config


def deep_merge_dicts(base: Dict, override: Dict) -> Dict:
    """Recursively merge dictionaries"""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters"""
    # Check required sections
    required_sections = ['training', 'ddpg', 'rewards', 'networks']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate learning rates
    if config['ddpg']['actor_lr'] <= 0 or config['ddpg']['critic_lr'] <= 0:
        raise ValueError("Learning rates must be positive")
    
    # Validate reward parameters
    if config['rewards']['failure_penalty'] > 0:
        raise ValueError("Failure penalty should be negative")
    
    # Validate network architectures
    if not config['networks']['actor']['hidden_dims']:
        raise ValueError("Actor network must have hidden layers")
    
    if not config['networks']['critic']['hidden_dims']:
        raise ValueError("Critic network must have hidden layers")
    
    print("✅ Configuration validation passed")


def print_config(config: Dict[str, Any]) -> None:
    """Pretty print configuration"""
    print("=" * 60)
    print("DDPG CONFIGURATION")
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


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """Save configuration to JSON file"""
    import json
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {filepath}")


def load_config_from_file(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    import json
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    validate_config(config)
    return config


# Quick access to common configurations
def get_debug_config(task: str = 'harvester') -> Dict[str, Any]:
    """Get configuration optimized for debugging (fast training, detailed logging)"""
    debug_overrides = {
        'training': {
            'max_episodes': 100,
            'eval_frequency': 10,
            'log_frequency': 1,
        },
        'ddpg': {
            'batch_size': 16,
            'buffer_size': 1000,
            'noise_std': 0.5,
        },
        'debug': {
            'log_program_samples': True,
            'log_latent_stats': True,
            'verbose_evaluation': True,
        }
    }
    
    return get_config(task, debug_overrides)


def get_production_config(task: str = 'harvester') -> Dict[str, Any]:
    """Get configuration optimized for production training"""
    production_overrides = {
        'training': {
            'max_episodes': 5000,
            'eval_frequency': 100,
        },
        'ddpg': {
            'batch_size': 64,
            'buffer_size': 100000,
        },
        'debug': {
            'log_program_samples': False,
            'save_execution_traces': False,
        }
    }
    
    return get_config(task, production_overrides)


def get_curriculum_config(task: str = 'harvester') -> Dict[str, Any]:
    """Get configuration with curriculum learning enabled"""
    curriculum_overrides = {
        'curriculum': CURRICULUM_CONFIG.copy()
    }
    curriculum_overrides['curriculum']['enabled'] = True
    
    return get_config(task, curriculum_overrides)


# Example usage
if __name__ == "__main__":
    # Test configuration system
    print("Testing DDPG Configuration System...")
    
    # Test basic config
    config = get_config('harvester')
    print_config(config)
    
    # Test task-specific config
    maze_config = get_config('randomMaze')
    print(f"Maze config macro steps: {maze_config['ddpg']['macro_steps']}")
    
    # Test custom overrides
    custom_config = get_config('harvester', {
        'ddpg': {'actor_lr': 1e-3},
        'rewards': {'success_bonus': 5.0}
    })
    print(f"Custom actor LR: {custom_config['ddpg']['actor_lr']}")
    print(f"Custom success bonus: {custom_config['rewards']['success_bonus']}")
    
    # Test debug config
    debug_config = get_debug_config('fourCorners')
    print(f"Debug episodes: {debug_config['training']['max_episodes']}")
    
    print("✅ Configuration system tests passed!")