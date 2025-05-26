#!/usr/bin/env python3
"""
HPRL VAE Training Script

This script formally trains the VAE component of the HPRL framework.

Usage:
    python train_vae.py [--epochs N] [--batch-size N] [--wandb] [--dummy]
"""

import argparse
import os
import sys
import torch
import logging
from datetime import datetime

# Add project modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.vae_trainer import HPRLVAETrainer
from utils.config import config


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train HPRL VAE Component")
    
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Training batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--beta', type=float, default=None, help='Beta coefficient for KL loss')
    parser.add_argument('--lambda-behavior', type=float, default=None, help='Lambda coefficient for behavior loss')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--dummy', action='store_true', help='Use dummy data for testing')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/vae', help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cuda/cpu)')
    
    return parser.parse_args()


def setup_device(device_arg: str):
    """Setup training device"""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    
    return device


def update_config_from_args(config, args):
    """Update configuration with ANTI-COLLAPSE settings"""
    if args.epochs is not None:
        config['train']['max_epoch'] = args.epochs
    else:
        config['train']['max_epoch'] = 150  # More epochs
    
    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
    else:
        config['train']['batch_size'] = 128  # Smaller batches
    
    if args.lr is not None:
        config['optimizer']['params']['lr'] = args.lr
    else:
        config['optimizer']['params']['lr'] = 5e-5  # Much lower LR
    
    if args.beta is not None:
        config['loss']['latent_loss_coef'] = args.beta
    else:
        config['loss']['latent_loss_coef'] = 0.5  # Much higher beta
    
    if args.lambda_behavior is not None:
        config['loss']['condition_loss_coef'] = args.lambda_behavior
    else:
        config['loss']['condition_loss_coef'] = 0.0  # Start with 0
    
    return config


def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup device
    device = setup_device(args.device)
    print(f"Using device: {device}")
    
    # Setup checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Use existing config and update with args
    training_config = config.copy()
    training_config = update_config_from_args(training_config, args)
    
    # Print configuration
    print("\nTraining Configuration:")
    print(f"  Epochs: {training_config['train']['max_epoch']}")
    print(f"  Batch Size: {training_config['train']['batch_size']}")
    print(f"  Learning Rate: {training_config['optimizer']['params']['lr']}")
    print(f"  Beta (KL): {training_config['loss']['latent_loss_coef']}")
    print(f"  Lambda (Behavior): {training_config['loss']['condition_loss_coef']}")
    print(f"  Use Dummy Data: {args.dummy}")
    print(f"  Wandb: {args.wandb}")
    # ADDED: Print anti-collapse settings
    print(f"\n🛡️ ANTI-COLLAPSE SETTINGS:")
    print(f"  Learning Rate: {training_config['optimizer']['params']['lr']}")
    print(f"  Beta (KL weight): {training_config['loss']['latent_loss_coef']}")
    print(f"  Batch Size: {training_config['train']['batch_size']}")
    print(f"  Max Epochs: {training_config['train']['max_epoch']}")
    print(f"  🎯 Target: Keep KL loss > 0.1")
    print()
    
    # Create trainer
    print("Creating VAE trainer...")
    trainer = HPRLVAETrainer(
        config=training_config,
        device=device,
        use_wandb=args.wandb,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Use dummy data if requested
    if args.dummy:
        print("Using dummy data for fast testing...")
        trainer._use_dummy_data = True
        trainer._setup_data_loaders()
    
    # Start training
    try:
        print("Starting VAE training...")
        trainer.train(resume_from=args.resume)
        
        print("Training completed successfully!")
        
        # Generate sample programs
        print("Generating sample programs...")
        samples = trainer.generate_samples(num_samples=5)
        
        print("\nGenerated Sample Programs:")
        for i, sample in enumerate(samples, 1):
            print(f"  {i}: {sample}")
        
        print(f"\nCheckpoints saved in: {args.checkpoint_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()