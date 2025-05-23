"""
Quick test script for VAE Trainer using dummy data

This script tests the VAE trainer functionality without requiring
the large Karel dataset, making it much faster for development and testing.
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from vae_trainer import HPRLVAETrainer
    from utils.config import config
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all modules are properly implemented")
    sys.exit(1)

def test_vae_trainer():
    """Test VAE trainer with dummy data"""
    print("=" * 60)
    print("Testing VAE Trainer with Dummy Data")
    print("=" * 60)
    
    # Setup minimal configuration for testing
    test_config = {
        'seed': 42,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        
        # Training config
        'train': {
            'max_epoch': 2,
            'batch_size': 4,
            'shuffle': True
        },
        
        # Validation config
        'valid': {
            'batch_size': 4,
            'shuffle': False
        },
        
        # Data loader config
        'data_loader': {
            'num_workers': 0,
            'pin_memory': False,
            'drop_last': True
        },
        
        # Network config
        'net': {
            'num_rnn_encoder_units': 256,
            'dropout': 0.0,
            'rnn_type': 'GRU',
            'tanh_after_sample': True
        },
        
        # DSL config
        'dsl': {
            'max_program_len': 12,
            'num_agent_actions': 6
        },
        
        # Loss config
        'loss': {
            'latent_loss_coef': 1.0,
            'condition_loss_coef': 1.0
        },
        
        # Optimizer config
        'optimizer': {
            'params': {
                'lr': 5e-4
            },
            'scheduler': {
                'step_size': 10,
                'gamma': 0.95
            }
        },
        
        # RL config (for compatibility)
        'rl': {
            'envs': {
                'executable': {
                    'max_demo_length': 20
                }
            }
        },
        
        # Additional required keys
        'use_simplified_dsl': False,
        'max_demo_length': 20
    }
    
    print("Creating VAE Trainer...")
    
    try:
        # Create trainer
        trainer = HPRLVAETrainer(
            config=test_config,
            device=test_config['device'],
            use_wandb=False,
            checkpoint_dir='./test_checkpoints'
        )
        
        # Force dummy data for fast testing
        trainer._use_dummy_data = True
        trainer._setup_data_loaders()
        
        print("✓ VAE Trainer created successfully")
        print(f"  Device: {trainer.device}")
        print(f"  Vocab size: {trainer.vocab_size}")
        print(f"  Latent dim: {trainer.latent_dim}")
        print(f"  Hidden size: {trainer.hidden_size}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        
        # Get a batch from data loader
        batch = next(iter(trainer.train_loader))
        programs, program_ids, masks, states, target_actions, action_lengths = batch
        
        # Move to device
        programs = programs.to(trainer.device)
        states = states.to(trainer.device)
        target_actions = target_actions.to(trainer.device)
        action_lengths = action_lengths.to(trainer.device)
        program_lengths = masks.sum(dim=1).squeeze(-1).to(trainer.device)
        
        print(f"  Batch shapes:")
        print(f"    Programs: {programs.shape}")
        print(f"    States: {states.shape}")
        print(f"    Target actions: {target_actions.shape}")
        print(f"    Program lengths: {program_lengths.shape}")
        
        # Forward pass
        results = trainer.forward_pass(
            programs, program_lengths, states, target_actions, action_lengths
        )
        
        print(f"✓ Forward pass completed")
        print(f"  Total loss: {results['total_loss'].item():.4f}")
        print(f"  Recon loss: {results['recon_loss'].item():.4f}")
        print(f"  KL loss: {results['kl_loss'].item():.4f}")
        print(f"  Behavior loss: {results['behavior_loss'].item():.4f}")
        
        # Test backward pass
        print("\nTesting backward pass...")
        results['total_loss'].backward()
        print("✓ Backward pass completed")
        
        # Test training for one epoch
        print("\nTesting training epoch...")
        train_metrics = trainer.train_epoch()
        print(f"✓ Training epoch completed")
        print(f"  Average total loss: {train_metrics.get('total_loss', 0):.4f}")
        
        # Test validation
        print("\nTesting validation...")
        val_metrics = trainer.validate_epoch()
        print(f"✓ Validation completed")
        print(f"  Validation loss: {val_metrics.get('total_loss', 0):.4f}")
        
        # Test sample generation
        print("\nTesting sample generation...")
        samples = trainer.generate_samples(num_samples=3)
        print("✓ Sample generation completed")
        print("Generated samples:")
        for i, sample in enumerate(samples):
            print(f"  {i+1}: {sample}")
        
        # Test checkpoint saving
        print("\nTesting checkpoint saving...")
        trainer.save_checkpoint(0, is_best=True)
        print("✓ Checkpoint saved")
        
        print("\n🎉 All tests passed! VAE Trainer is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quick_training():
    """Test a quick training run"""
    print("\n" + "=" * 60)
    print("Testing Quick Training Run")
    print("=" * 60)
    
    # Minimal config for very fast training
    quick_config = {
        'seed': 42,
        'train': {'max_epoch': 1, 'batch_size': 2, 'shuffle': True},
        'valid': {'batch_size': 2, 'shuffle': False},
        'data_loader': {'num_workers': 0, 'pin_memory': False, 'drop_last': True},
        'net': {'num_rnn_encoder_units': 64, 'dropout': 0.0, 'rnn_type': 'GRU', 'tanh_after_sample': True},
        'dsl': {'max_program_len': 8, 'num_agent_actions': 6},
        'loss': {'latent_loss_coef': 0.1, 'condition_loss_coef': 0.1},
        'optimizer': {'params': {'lr': 1e-3}, 'scheduler': {'step_size': 10, 'gamma': 0.95}},
        'rl': {'envs': {'executable': {'max_demo_length': 10}}},
        'use_simplified_dsl': False,
        'max_demo_length': 10
    }
    
    try:
        trainer = HPRLVAETrainer(
            config=quick_config,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_wandb=False,
            checkpoint_dir='./quick_test_checkpoints'
        )
        
        # Use dummy data
        trainer._use_dummy_data = True
        trainer._setup_data_loaders()
        
        print("Starting quick training run...")
        trainer.train()
        
        print("✓ Quick training completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Quick training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting VAE Trainer Tests")
    print("This uses dummy data for fast testing without requiring the Karel dataset")
    
    # Run tests
    test1_passed = test_vae_trainer()
    test2_passed = test_quick_training()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if test1_passed and test2_passed:
        print("🎉 All tests passed! VAE Trainer is ready for use.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)