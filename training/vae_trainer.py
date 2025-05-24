"""
Modified VAE Trainer for HPRL - Using ProgramVAE instead of basic VAE

Key changes:
1. Replace VAE with ProgramVAE
2. Simplify model architecture (ProgramVAE handles everything)
3. Update forward pass to use ProgramVAE's integrated approach
4. Simplify loss computation using ProgramVAE's built-in loss method
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae import ProgramVAE  # ✅ Use ProgramVAE instead of VAE
from models.program_executor import ProgramExecutor
from training.data_loader import make_datasets, ProgramDataset
from dsl.karel_dsl import get_DSL_option_v2
from dsl.tokens import get_vocab_size, get_padding_index
from environments.karel_env import KarelEnvironment
from utils.config import config


class HPRLVAETrainer:
    """
    VAE Trainer for HPRL - Modified to use ProgramVAE
    
    Implements the training procedure with ProgramVAE that includes
    both program reconstruction and behavioral consistency.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = 'cuda',
        use_wandb: bool = False,
        checkpoint_dir: str = './checkpoints'
    ):
        """Initialize VAE Trainer with ProgramVAE"""
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize DSL and vocabulary
        self.dsl = get_DSL_option_v2(seed=config.get('seed', 42))
        self.vocab_size = get_vocab_size()
        self.padding_idx = get_padding_index()
        
        # Model parameters from config
        self.embedding_dim = 64
        self.hidden_size = config.get('net', {}).get('num_rnn_encoder_units', 256)
        self.latent_dim = 64  # Final compressed latent dimension
        self.max_program_length = config.get('dsl', {}).get('max_program_len', 12) + 1
        self.max_demo_length = config.get('rl', {}).get('envs', {}).get('executable', {}).get('max_demo_length', 100)
        
        # Training parameters
        self.beta = config.get('loss', {}).get('latent_loss_coef', 1.0)
        self.lambda_behavior = config.get('loss', {}).get('condition_loss_coef', 1.0)
        self.num_epochs = config.get('train', {}).get('max_epoch', 100)
        self.batch_size = config.get('train', {}).get('batch_size', 256)
        self.learning_rate = config.get('optimizer', {}).get('params', {}).get('lr', 5e-4)
        
        # Set num_agent_actions in config if not present
        if 'dsl' not in self.config:
            self.config['dsl'] = {}
        if 'num_agent_actions' not in self.config['dsl']:
            self.config['dsl']['num_agent_actions'] = len(self.dsl.action_functions) + 1
        
        # Add missing config keys that data_loader expects
        self._setup_config_defaults()
        
        # Initialize models - MAJOR CHANGE HERE
        self._build_models()
        
        # Initialize optimizers
        self._setup_optimizers()
        
        # Initialize data loaders
        self._setup_data_loaders()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
        self.logger.info("VAE Trainer initialized successfully with ProgramVAE")
    
    def _setup_config_defaults(self):
        """Setup default config values for compatibility"""
        self.config['use_simplified_dsl'] = self.config.get('use_simplified_dsl', False)
        if 'dsl_tokens' not in self.config:
            self.config['dsl_tokens'] = self.dsl.int2token
        if 'prl_tokens' not in self.config:
            self.config['prl_tokens'] = self.dsl.int2token
        if 'dsl2prl_mapping' not in self.config:
            self.config['dsl2prl_mapping'] = {token: token for token in self.dsl.int2token}
        if 'max_demo_length' not in self.config:
            self.config['max_demo_length'] = self.max_demo_length
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.checkpoint_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('VAETrainer')
    
    def _build_models(self):
        """
        Build models using ProgramVAE - MAJOR SIMPLIFICATION
        
        ProgramVAE includes:
        - VAE for program reconstruction
        - ConditionPolicy for behavior prediction
        - Integrated loss computation
        """
        self.logger.info("Building ProgramVAE model...")
        
        # ✅ MAIN CHANGE: Use ProgramVAE instead of separate components
        self.program_vae = ProgramVAE(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            latent_dim=self.latent_dim,
            state_shape=(8, 8, 8),  # Karel environment shape
            num_actions=self.config['dsl']['num_agent_actions'],
            max_program_length=self.max_program_length,
            max_demo_length=self.max_demo_length,
            dropout=self.config['net']['dropout'],
            rnn_type=self.config['net']['rnn_type']
        ).to(self.device)
        
        # Keep program executor for generating execution traces if needed
        self.program_executor = ProgramExecutor(
            vocab_size=self.vocab_size,
            max_program_length=self.max_program_length,
            max_execution_steps=100,
            device=self.device
        )
        
        self.logger.info(f"ProgramVAE built - Total params: {sum(p.numel() for p in self.program_vae.parameters()):,}")
    
    def _setup_optimizers(self):
        """Setup optimizer for ProgramVAE - Much simpler now"""
        # ✅ SIMPLIFIED: Only need to optimize ProgramVAE parameters
        self.optimizer = optim.Adam(
            self.program_vae.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        scheduler_config = self.config.get('optimizer', {}).get('scheduler', {})
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=scheduler_config.get('step_size', 10),
            gamma=scheduler_config.get('gamma', 0.95)
        )
        
        self.logger.info("Optimizer setup complete")
    
    def _setup_data_loaders(self):
        """Setup data loaders for training and validation"""
        self.logger.info("Setting up data loaders...")
        
        # For testing, use dummy data
        if hasattr(self, '_use_dummy_data') and self._use_dummy_data:
            self.logger.info("Using dummy data for testing...")
            self._create_dummy_datasets()
            return
        
        # Try to find real data
        possible_datadirs = [
            "../data/karel_dataset_option_L30_1m_cover_branch",
            "./data/karel_dataset_option_L30_1m_cover_branch", 
            "data/karel_dataset_option_L30_1m_cover_branch",
            "../data",
            "./data"
        ]
        
        datadir = None
        for path in possible_datadirs:
            if os.path.exists(path):
                datadir = path
                break
        
        if datadir is None:
            self.logger.warning("No data directory found. Creating dummy datasets.")
            self._create_dummy_datasets()
            return

        train_dataset, val_dataset, test_dataset = make_datasets(
            datadir=datadir,
            config=self.config,
            num_program_tokens=self.vocab_size,
            num_agent_actions=self.config.get('dsl', {}).get('num_agent_actions', 6),
            device=self.device,
            dsl=self.dsl
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.config.get('train', {}).get('shuffle', True),
            num_workers=self.config.get('data_loader', {}).get('num_workers', 0),
            pin_memory=self.config.get('data_loader', {}).get('pin_memory', False),
            drop_last=self.config.get('data_loader', {}).get('drop_last', True)
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('valid', {}).get('batch_size', 256),
            shuffle=self.config.get('valid', {}).get('shuffle', True),
            num_workers=self.config.get('data_loader', {}).get('num_workers', 0),
            pin_memory=self.config.get('data_loader', {}).get('pin_memory', False),
            drop_last=False
        )
        
        self.logger.info(f"Data loaders created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    def _create_dummy_datasets(self):
        """Create dummy datasets for testing"""
        self.logger.warning("Creating dummy datasets for testing...")
        
        from torch.utils.data import TensorDataset
        
        # Generate dummy data - smaller for testing
        num_samples = 100
        dummy_programs = torch.randint(0, self.vocab_size-1, (num_samples, self.max_program_length))
        dummy_program_ids = torch.arange(num_samples)
        dummy_lengths = torch.randint(5, self.max_program_length, (num_samples,))
        dummy_masks = torch.ones(num_samples, self.max_program_length, 1, dtype=torch.bool)
        
        # Create proper masks based on lengths
        for i, length in enumerate(dummy_lengths):
            dummy_masks[i, length:] = False
        
        # Create dummy execution data
        num_demos = 3
        demo_length = 10
        dummy_states = torch.randn(num_samples, num_demos, demo_length, 8, 8, 8)
        dummy_actions = torch.randint(0, 5, (num_samples, num_demos, demo_length-1), dtype=torch.long)
        dummy_action_lengths = torch.randint(1, demo_length-1, (num_samples, num_demos), dtype=torch.long)
        
        # Split into train/val
        train_size = int(0.8 * num_samples)
        
        train_dataset = TensorDataset(
            dummy_programs[:train_size], 
            dummy_program_ids[:train_size], 
            dummy_masks[:train_size],
            dummy_states[:train_size], 
            dummy_actions[:train_size], 
            dummy_action_lengths[:train_size]
        )
        
        val_dataset = TensorDataset(
            dummy_programs[train_size:], 
            dummy_program_ids[train_size:], 
            dummy_masks[train_size:],
            dummy_states[train_size:], 
            dummy_actions[train_size:], 
            dummy_action_lengths[train_size:]
        )
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.logger.info(f"Dummy data loaders created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    def forward_pass(
        self, 
        programs: torch.Tensor, 
        program_lengths: torch.Tensor,
        states: torch.Tensor,
        target_actions: torch.Tensor,
        action_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        ✅ SIMPLIFIED FORWARD PASS using ProgramVAE
        
        ProgramVAE handles all the complexity internally:
        - Program encoding/decoding
        - Latent sampling
        - Action prediction
        - Loss computation
        """
        # ✅ MAJOR SIMPLIFICATION: One call to ProgramVAE
        results = self.program_vae(
            programs=programs,
            program_lengths=program_lengths,
            states=states,
            target_actions=target_actions,
            target_programs=programs,  # Use same programs as targets for reconstruction
            deterministic=False,
            compute_policy=True
        )
        
        # ✅ SIMPLIFIED LOSS COMPUTATION using ProgramVAE's built-in method
        losses = self.program_vae.loss(
            programs=programs,
            program_lengths=program_lengths,
            target_programs=programs,
            states=states,
            target_actions=target_actions,
            beta=self.beta,
            action_weight=self.lambda_behavior
        )
        
        # Combine results and losses
        results.update(losses)
        
        return results
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch - Much simpler now"""
        self.program_vae.train()
        
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Unpack batch
            programs, program_ids, masks, states, target_actions, action_lengths = batch
            
            # Move to device and ensure correct data types
            programs = programs.to(self.device).long()
            states = states.to(self.device).float()
            target_actions = target_actions.to(self.device).long()
            action_lengths = action_lengths.to(self.device).long()
            
            # Get actual program lengths from masks
            program_lengths = masks.sum(dim=1).squeeze(-1).to(self.device).long()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # ✅ SIMPLIFIED: Forward pass using ProgramVAE
            results = self.forward_pass(
                programs, program_lengths, states, target_actions, action_lengths
            )
            
            # Backward pass
            results['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.program_vae.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            for key, value in results.items():
                if isinstance(value, torch.Tensor) and 'loss' in key:
                    epoch_metrics[key] += value.item()
            
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            current_loss = results['total_loss'].item()
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'recon': f'{results["recon_loss"].item():.4f}',
                'kl': f'{results["kl_loss"].item():.4f}',
                'action': f'{results["action_loss"]:.4f}'
            })
        
        # Average metrics
        if num_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
        
        return dict(epoch_metrics)
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.program_vae.eval()
        
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Unpack batch
                programs, program_ids, masks, states, target_actions, action_lengths = batch
                
                # Move to device and ensure correct data types
                programs = programs.to(self.device).long()
                states = states.to(self.device).float()
                target_actions = target_actions.to(self.device).long()
                action_lengths = action_lengths.to(self.device).long()
                
                # Get actual program lengths
                program_lengths = masks.sum(dim=1).squeeze(-1).to(self.device).long()
                
                # Forward pass
                results = self.forward_pass(
                    programs, program_lengths, states, target_actions, action_lengths
                )
                
                # Update metrics
                for key, value in results.items():
                    if isinstance(value, torch.Tensor) and 'loss' in key:
                        epoch_metrics[key] += value.item()
                
                num_batches += 1
        
        # Average metrics
        if num_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
        
        return dict(epoch_metrics)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint - Simplified"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'program_vae_state_dict': self.program_vae.state_dict(),  # ✅ Single model
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics)
        }
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch}")
        
        # Save latest model
        latest_path = os.path.join(self.checkpoint_dir, 'latest_model.pt')
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint - Simplified"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.program_vae.load_state_dict(checkpoint['program_vae_state_dict'])  # ✅ Single model
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop"""
        if resume_from:
            self.load_checkpoint(resume_from)
        
        self.logger.info("Starting ProgramVAE training...")
        self.logger.info(f"Training for {self.num_epochs} epochs")
        self.logger.info(f"Beta (KL weight): {self.beta}")
        self.logger.info(f"Lambda (action weight): {self.lambda_behavior}")
        
        try:
            for epoch in range(self.current_epoch, self.num_epochs):
                self.current_epoch = epoch
                
                # Training
                train_metrics = self.train_epoch()
                
                # Validation
                val_metrics = self.validate_epoch()
                
                # Update learning rate
                self.scheduler.step()
                
                # Log epoch results
                self.logger.info(f"Epoch {epoch} - Train Loss: {train_metrics.get('total_loss', 0):.4f}, "
                               f"Val Loss: {val_metrics.get('total_loss', 0):.4f}")
                
                # Save metrics
                for key, value in train_metrics.items():
                    self.train_metrics[key].append(value)
                for key, value in val_metrics.items():
                    self.val_metrics[key].append(value)
                
                # Save checkpoint
                val_loss = val_metrics.get('total_loss', float('inf'))
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                if epoch % self.config.get('save_interval', 10) == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        self.logger.info("Training completed")
        self.save_checkpoint(self.current_epoch, False)
    
    def generate_samples(self, num_samples: int = 10) -> List[str]:
        """Generate sample programs from the trained ProgramVAE"""
        self.program_vae.eval()
        
        generated_programs = []
        
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            
            # Generate programs using ProgramVAE
            predicted_tokens = self.program_vae.generate_program(z, deterministic=True)
            
            for i in range(num_samples):
                tokens = predicted_tokens[i].cpu().numpy()
                # Remove padding
                tokens = tokens[tokens != self.padding_idx]
                program_str = self.dsl.intseq2str(tokens.tolist())
                generated_programs.append(program_str)
        
        return generated_programs


# Example usage
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message=".*NumPy.*")
    
    # Setup configuration
    test_config = config.copy()
    test_config['train']['max_epoch'] = 2
    test_config['train']['batch_size'] = 4
    test_config['valid']['batch_size'] = 4
    
    # Create trainer
    trainer = HPRLVAETrainer(
        config=test_config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_wandb=False,
        checkpoint_dir='./test_checkpoints'
    )
    
    # Force use of dummy data for testing
    trainer._use_dummy_data = True
    trainer._setup_data_loaders()
    
    print("ProgramVAE Trainer created successfully!")
    
    try:
        trainer.train()
        print("Training test completed!")
        
        # Test sample generation
        samples = trainer.generate_samples(num_samples=3)
        print("\nGenerated samples:")
        for i, sample in enumerate(samples):
            print(f"  {i+1}: {sample}")
            
    except Exception as e:
        print(f"Error during training test: {e}")
        import traceback
        traceback.print_exc()