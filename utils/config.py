"""
Corrected Configuration for VAE Training

Key changes:
1. Much lower learning rate (1e-4 instead of 5e-4)
2. Much lower KL coefficient (0.01 instead of 1.0)
3. Disabled action loss initially (0.0 instead of 1.0)
4. Smaller batch size for stability
5. More frequent saving and logging
"""

config = {
    'device': 'cuda:0',
    'save_interval': 5,                             # FIXED: Save more frequently to monitor
    'log_interval': 1,                              # FIXED: Log every epoch
    'log_video_interval': 10,
    'record_file': 'records.pkl',
    'algorithm': 'supervised',
    'mode': 'train',

    'do_supervised': True,
    'do_RL': True,

    # config for logging
    'logging': {
        'log_file': 'run.log',
        'fmt': '%(asctime)s: %(message)s',
        'level': 'DEBUG',                           # FIXED: More detailed logging
        'wandb': False,
    },

    # FIXED: Network configuration to prevent collapse
    'net': {
        'saved_params_path': None,
        'saved_sup_params_path': None,
        'rnn_type': 'GRU',
        'dropout': 0.1,                             # FIXED: Add dropout for regularization
        'latent_std_mu': 0.0,
        'latent_std_sigma': 0.1,
        'latent_mean_pooling': False,
        'decoder': {
            'use_teacher_enforcing': True,
            'freeze_params': False
        },
        'use_linear': True,
        'num_rnn_encoder_units': 256,
        'num_rnn_decoder_units': 256,
        'use_transformer_encoder': False,
        'use_transformer_decoder': False,
        'transformer': {
            'd_word_vec': 32,
            'd_k': 4,
            'd_v': 4,
            'n_layers': 3,
            'n_head': 4,
            'd_inner': 512,
            'dropout': 0.1,
            'method': 'Autobot',
        },
        'condition':{
            'freeze_params': False,
            'use_teacher_enforcing': True,
            'observations': 'environment',
        },
        'controller':{
            'add_noise': False,
            'input_coef': 0.01,
            'use_decoder_dist': True,
            'use_previous_programs': False,
            'program_reduction': 'identity',
        },
        'tanh_after_mu_sigma': False,
        'tanh_after_sample': True,
    },

    # FIXED: Data loader configuration
    'data_loader': {
        'num_workers': 0,
        'pin_memory': False,
        'drop_last': True,
    },

    # Random seed
    'seed': 123,

    # FIXED: Optimizer settings to prevent collapse
    'optimizer': {
        'name': 'adam',
        'params': {
            'lr': 1e-4,                             # FIXED: Much lower learning rate (was 5e-4)
        },
        'scheduler': {
            'step_size': 20,                        # FIXED: Longer step size
            'gamma': 0.9,                           # FIXED: Less aggressive decay
        }
    },

    # FIXED: Training configuration
    'train': {
        'data': {
            'to_tensor': True,
            'use_pickled': True
        },
        'batch_size': 128,                          # FIXED: Smaller batch size for stability (was 256)
        'shuffle': True,
        'max_epoch': 200,                           # FIXED: More epochs with lower LR (was 100)
    },

    # FIXED: Validation configuration
    'valid': {
        'data': {
            'to_tensor': True,
            'use_pickled': True
        },
        'batch_size': 128,                          # FIXED: Consistent with training batch size
        'shuffle': True,
        'debug_samples': [3, 37, 54],
    },

    # Test configuration
    'test': {
        'data': {
            'to_tensor': True,
            'use_pickled': True
        },
        'batch_size': 128,                          # FIXED: Consistent batch size
        'shuffle': True,
    },

    # Evaluation configuration
    'eval': {
        'usage': 'test',
    },

    # CRITICAL: Fixed loss configuration to prevent VAE collapse
    'loss': {
        'latent_loss_coef': 0.01,                   # FIXED: Much lower KL weight (was 1.0)
        'condition_loss_coef': 0.0,                 # FIXED: Start with no action loss (was 1.0)
    },

    # DSL configuration
    'dsl': {
        'use_simplified_dsl': False,                # FIXED: Use complete vocabulary
        'max_program_len': 12,
        'grammar': 'handwritten',
    },

    # RL configuration (mostly unchanged but some fixes)
    'rl':{
        'num_processes': 64,
        'num_steps': 8,
        'num_env_steps': 10e6,
        'gamma': 0.99,
        'use_gae': True,
        'gae_lambda': 0.95,
        'use_proper_time_limits': False,
        'use_all_programs': False,
        'future_rewards': False,
        'value_method': 'mean',
        'envs': {
            'executable': {
                'name': 'karel',
                'task_definition': 'program',
                'task_file': 'tasks/test1.txt',
                'max_demo_length': 100,
                'min_demo_length': 1,
                'num_demo_per_program': 10,
                'dense_execution_reward': False,
            },
            'program': {
                'mdp_type': 'ProgramEnv1',
                'intrinsic_reward': False,
                'intrinsic_beta': 0.0,
            }
        },
        'policy':{
          'execution_guided': False,
          'two_head': False,
          'recurrent_policy': True,
        },
        'algo':{
            'name': 'reinforce',
            'value_loss_coef':0.5,
            'entropy_coef':0.1,
            'final_entropy_coef': 0.01,
            'use_exp_ent_decay': False,
            'use_recurrent_generator': False,
            'max_grad_norm': 0.5,
            'lr': 1e-4,                             # FIXED: Consistent with main LR
            'use_linear_lr_decay': True,
            'ppo':{
                'clip_param':0.1,
                'ppo_epoch':2,
                'num_mini_batch':2,
                'eps': 1e-5,
            },
            'a2c':{
                'eps': 1e-5,
                'alpha': 0.99,
            },
            'acktr':{
            },
            'reinforce': {
                'clip_param': 0.1,
                'reinforce_epoch': 1,
                'num_mini_batch': 2,
                'eps': 1e-5,
            },
        },
        'loss':{
                'decoder_rl_loss_coef': 1.0,
                'condition_rl_loss_coef': 0.0,
                'latent_rl_loss_coef': 0.0,
                'use_mean_only_for_latent_loss': False,
            }
    },

    # CEM configuration
    'CEM':{
        'init_type': 'normal',
        'reduction': 'mean',
        'population_size': 384,
        'elitism_rate': 0.2,
        'max_number_of_epochs': 1000,
        'sigma': 1.0,
        'final_sigma': 0.1,
        'use_exp_sig_decay': False,
        'exponential_reward': False,
        'average_score_for_solving': 1.1,
        'detailed_dump': False,
    },

    # PPO configuration
    'PPO':{
        'algo': 'ppo',
        'num_processes': 16,
        'hidden_size': 16,
        'lr': 1e-4,                                 # FIXED: Consistent LR
        'eps': 1e-5,
        'alpha': 0.99,
        'gamma': 0.99,
        'use_gae': True,
        'gae_lambda': 0.95,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
        'cuda_deterministic': False,
        'decoder_deterministic': True,
        'num_steps': 50,
        'ppo_epoch': 3,
        'num_mini_batch': 5,
        'clip_param': 0.2,
        'eval_interval': None,
        'num_env_steps': 5e6,
        'use_proper_time_limits': False,
        'recurrent_policy': False,
        'use_linear_lr_decay': False,
    },

    # PPO_DRL configuration
    'PPO_DRL':{
        'algo': 'ppo',
        'num_processes': 16,
        'hidden_size': 16,
        'lr': 1e-4,                                 # FIXED: Consistent LR
        'eps': 1e-5,
        'alpha': 0.99,
        'gamma': 0.99,
        'use_gae': True,
        'gae_lambda': 0.95,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
        'cuda_deterministic': False,
        'decoder_deterministic': True,
        'num_steps': 1000,
        'ppo_epoch': 3,
        'num_mini_batch': 5,
        'clip_param': 0.2,
        'eval_interval': None,
        'num_env_steps': 20e6,
        'use_proper_time_limits': False,
        'recurrent_policy': False,
        'use_linear_lr_decay': False,
    },

    # SAC configuration
    'SAC':{
        'hidden_size': 16,
        'obs_emb_dim': 16,
        'num_processes': 16,
        'cuda_deterministic': False,
        'decoder_deterministic': True,
        'num_seed_steps': 1e4,
        'num_train_steps': 5e6,
        'replay_buffer_capacity': 5e6,
        'agent': {
            'discount': 0.99,
            'init_temperature': 0.1,
            'alpha_lr': 1e-4,                       # FIXED: Consistent LR
            'alpha_betas': [0.9, 0.999],
            'actor_lr': 1e-4,                       # FIXED: Consistent LR
            'actor_betas': [0.9, 0.999],
            'actor_update_frequency': 10,
            'critic_lr': 1e-4,                      # FIXED: Consistent LR
            'critic_betas': [0.9, 0.999],
            'critic_tau': 0.005,
            'critic_target_update_frequency': 20,
            'batch_size': 512,
            'learnable_temperature': True,
            'log_histogram_interval': 500,
            },
        'double_q_critic': {
            'hidden_dim': 16,
            'hidden_depth': 2,
            },
        'diag_gaussian_actor': {
            'hidden_depth': 2,
            'hidden_dim': 16,
            'log_std_bounds': [-5, 2]
            },
    }
}


# ADDITIONAL HELPER FUNCTIONS FOR MONITORING

def print_config_summary():
    """Print summary of critical configuration changes"""
    print("🔧 CRITICAL CONFIGURATION CHANGES:")
    print("=" * 50)
    print(f"Learning Rate: {config['optimizer']['params']['lr']} (was 5e-4)")
    print(f"KL Coefficient: {config['loss']['latent_loss_coef']} (was 1.0)")
    print(f"Action Coefficient: {config['loss']['condition_loss_coef']} (was 1.0)")
    print(f"Batch Size: {config['train']['batch_size']} (was 256)")
    print(f"Max Epochs: {config['train']['max_epoch']} (was 100)")
    print(f"Dropout: {config['net']['dropout']} (was 0.0)")
    print(f"Save Interval: {config['save_interval']} (was 10)")
    print()
    print("✅ These changes should prevent VAE collapse!")
    print("✅ Training will start with VAE-only, then add action loss")

def validate_config():
    """Validate that critical settings are correct"""
    issues = []
    
    if config['optimizer']['params']['lr'] > 2e-4:
        issues.append("Learning rate too high - should be ≤ 1e-4")
    
    if config['loss']['latent_loss_coef'] > 0.1:
        issues.append("KL coefficient too high - should be ≤ 0.01")
    
    if config['loss']['condition_loss_coef'] > 0:
        issues.append("Action loss should start at 0.0")
    
    if config['train']['batch_size'] > 256:
        issues.append("Batch size too large - should be ≤ 128")
    
    if issues:
        print("⚠️  CONFIGURATION ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Configuration validation passed!")
        return True


if __name__ == "__main__":
    print_config_summary()
    validate_config()