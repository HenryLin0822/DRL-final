"""
VAE Trainer for HPRL

This module implements the VAE training loop for the Hierarchical Programmatic 
Reinforcement Learning framework, following the paper's methodology.

Key components:
1. Program reconstruction loss (L^P) - β-VAE objective
2. Latent behavior reconstruction loss (L^L) - behavioral smoothness
3. Compression encoder/decoder training
4. Integration with program executor for execution traces
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
import matplotlib.pyplot as plt

# Handle NumPy compatibility issue with wandb
import warnings
warnings.filterwarnings("ignore", message=".*NumPy.*")

# Optional wandb import

WANDB_AVAILABLE = False
print("Warning: wandb not available. Logging will be disabled.")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae import ProgramVAE, VAE
from models.program_executor import ProgramExecutor, NeuralProgramExecutor
from training.data_loader import make_datasets, ProgramDataset
from dsl.karel_dsl import get_DSL_option_v2
from dsl.tokens import get_vocab_size, get_padding_index
from environments.karel_env import KarelEnvironment
from utils.config import config


class HPRLVAETrainer:
    """
    VAE Trainer for HPRL
    
    Implements the training procedure described in Algorithm 1 of the paper,
    combining program reconstruction and latent behavior reconstruction losses.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = 'cuda',
        use_wandb: bool = False,
        checkpoint_dir: str = './checkpoints'
    ):
        """
        Initialize VAE Trainer
        
        Args:
            config: Training configuration dictionary
            device: Device to train on
            use_wandb: Whether to use Weights & Biases logging
            checkpoint_dir: Directory to save checkpoints
        """
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
        self.latent_dim = 64  # Compressed latent dimension
        self.uncompressed_latent_dim = 256  # Before compression
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
        self.config['use_simplified_dsl'] = self.config.get('use_simplified_dsl', False)
        if 'dsl_tokens' not in self.config:
            self.config['dsl_tokens'] = self.dsl.int2token
        if 'prl_tokens' not in self.config:
            self.config['prl_tokens'] = self.dsl.int2token  # Same as dsl_tokens for now
        if 'dsl2prl_mapping' not in self.config:
            self.config['dsl2prl_mapping'] = {token: token for token in self.dsl.int2token}
        if 'max_demo_length' not in self.config:
            self.config['max_demo_length'] = self.max_demo_length
        
        # Initialize models
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
        
        self.logger.info("VAE Trainer initialized successfully")
    
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
        
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project="hprl-vae-training",
                config=self.config,
                name=f"vae_training_{self.config['seed']}"
            )
        elif self.use_wandb and not WANDB_AVAILABLE:
            self.logger.warning("wandb requested but not available. Disabling wandb logging.")
            self.use_wandb = False
    
    def _build_models(self):
        """Build VAE models and related components"""
        self.logger.info("Building models...")
        
        # Main VAE model (uncompressed)
        self.vae_uncompressed = VAE(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            latent_dim=self.uncompressed_latent_dim,
            max_program_length=self.max_program_length,
            dropout=self.config['net']['dropout'],
            rnn_type=self.config['net']['rnn_type']
        ).to(self.device)
        
        # Compression encoder and decoder (f_ω and g_ψ from paper)
        self.compression_encoder = nn.Sequential(
            nn.Linear(self.uncompressed_latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, self.latent_dim)
        ).to(self.device)
        
        self.compression_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 128), 
            nn.Tanh(),
            nn.Linear(128, self.uncompressed_latent_dim)
        ).to(self.device)
        
        # Neural program executor (π from paper) for latent behavior reconstruction
        self.neural_executor = NeuralProgramExecutor(
            latent_dim=self.latent_dim,
            state_dim=8 * 8 * 8,  # Flattened Karel state
            hidden_dim=self.hidden_size,
            num_actions=self.config['dsl']['num_agent_actions'],
            max_sequence_length=self.max_demo_length - 1,
            dropout=self.config['net']['dropout']
        ).to(self.device)
        
        # Program executor for generating execution traces
        self.program_executor = ProgramExecutor(
            vocab_size=self.vocab_size,
            max_program_length=self.max_program_length,
            max_execution_steps=100,
            device=self.device
        )
        
        # Optional: Apply tanh after sampling if specified in config
        self.use_tanh_after_sample = self.config.get('net', {}).get('tanh_after_sample', True)
        
        self.logger.info(f"Models built - VAE params: {sum(p.numel() for p in self.vae_uncompressed.parameters()):,}")
    
    def _setup_optimizers(self):
        """Setup optimizers for all components"""
        # Combine all parameters for joint optimization (as in Algorithm 1)
        all_params = []
        all_params.extend(self.vae_uncompressed.parameters())
        all_params.extend(self.compression_encoder.parameters())
        all_params.extend(self.compression_decoder.parameters())
        all_params.extend(self.neural_executor.parameters())
        
        self.optimizer = optim.Adam(
            all_params,
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
        
        self.logger.info("Optimizers setup complete")
    
    def _setup_data_loaders(self):
        """Setup data loaders for training and validation"""
        self.logger.info("Setting up data loaders...")
        
        # For testing, always use dummy data to avoid long loading times
        if hasattr(self, '_use_dummy_data') and self._use_dummy_data:
            self.logger.info("Using dummy data for testing...")
            self._create_dummy_datasets()
            return
        
        # Try multiple possible data directories
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
            

    
    ''' def _create_dummy_datasets(self):
        """Create dummy datasets for testing when real data is not available"""
        self.logger.warning("Creating dummy datasets for testing...")
        
        from torch.utils.data import TensorDataset
        
        # Generate dummy data - much smaller for testing
        num_samples = 100  # Small dataset for fast testing
        dummy_programs = torch.randint(0, self.vocab_size-1, (num_samples, self.max_program_length))
        dummy_program_ids = torch.arange(num_samples)
        dummy_lengths = torch.randint(5, self.max_program_length, (num_samples,))
        dummy_masks = torch.ones(num_samples, self.max_program_length, 1, dtype=torch.bool)
        
        # Create proper masks based on lengths
        for i, length in enumerate(dummy_lengths):
            dummy_masks[i, length:] = False
        
        # Create dummy execution data
        num_demos = 3  # Reduced from 10 for faster testing
        demo_length = 10  # Reduced demo length
        dummy_states = torch.randn(num_samples, num_demos, demo_length, 8, 8, 8)
        dummy_actions = torch.randint(0, 5, (num_samples, num_demos, demo_length-1), dtype=torch.long)  # Ensure long type
        dummy_action_lengths = torch.randint(1, demo_length-1, (num_samples, num_demos), dtype=torch.long)  # Ensure long type
        
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
        
        self.logger.info(f"Dummy data loaders created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")'''
    
    def forward_pass(
        self, 
        programs: torch.Tensor, 
        program_lengths: torch.Tensor,
        states: torch.Tensor,
        target_actions: torch.Tensor,
        action_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass implementing the full HPRL VAE objective
        
        Args:
            programs: [batch_size, seq_len] - tokenized programs
            program_lengths: [batch_size] - actual program lengths
            states: [batch_size, num_demos, demo_len, H, W, C] - execution states
            target_actions: [batch_size, num_demos, demo_len-1] - target action sequences
            action_lengths: [batch_size, num_demos] - actual action sequence lengths
            
        Returns:
            Dictionary containing all loss components and intermediate results
        """
        batch_size = programs.size(0)
        
        # 1. Encode programs to uncompressed latent space
        mu_uncompressed, logvar_uncompressed = self.vae_uncompressed.encode(programs, program_lengths)
        
        # 2. Sample from uncompressed latent space
        z_uncompressed = self.vae_uncompressed.reparameterize(mu_uncompressed, logvar_uncompressed)
        
        # 3. Compress latent representations (f_ω from paper)
        z_compressed = self.compression_encoder(z_uncompressed)
        
        # Apply tanh if specified in config
        if self.use_tanh_after_sample:
            z_compressed = torch.tanh(z_compressed)
        
        # 4. Decompress for program reconstruction (g_ψ from paper)
        z_decompressed = self.compression_decoder(z_compressed)
        
        # 5. Decode programs (program reconstruction loss L^P)
        output_logits, output_log_probs = self.vae_uncompressed.decoder(
            z_decompressed, 
            target_programs=programs,
            deterministic=False
        )
        
        # 6. Compute program reconstruction loss (β-VAE objective)
        recon_loss = self._compute_program_reconstruction_loss(
            output_logits, programs, program_lengths
        )
        
        # 7. Compute KL divergence loss
        kl_loss = self.vae_uncompressed.kl_loss(mu_uncompressed, logvar_uncompressed)
        
        # 8. Compute latent behavior reconstruction loss (L^L from paper)
        behavior_loss = self._compute_latent_behavior_loss(
            z_compressed, states, target_actions, action_lengths
        )
        
        # 9. Combine losses according to Equation (3) from paper
        total_loss = recon_loss + self.beta * kl_loss + self.lambda_behavior * behavior_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'behavior_loss': behavior_loss,
            'z_compressed': z_compressed,
            'z_uncompressed': z_uncompressed,
            'output_logits': output_logits,
            'mu': mu_uncompressed,
            'logvar': logvar_uncompressed
        }
    
    def _compute_program_reconstruction_loss(
        self,
        output_logits: torch.Tensor,
        target_programs: torch.Tensor,
        program_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute program reconstruction loss (L^P from paper)
        
        Args:
            output_logits: [batch_size, seq_len, vocab_size] - predicted logits
            target_programs: [batch_size, seq_len] - target programs
            program_lengths: [batch_size] - actual program lengths
            
        Returns:
            Program reconstruction loss
        """
        batch_size, seq_len, vocab_size = output_logits.shape
        target_seq_len = target_programs.size(1)
        
        # Handle length mismatch
        min_len = min(seq_len, target_seq_len)
        output_logits_truncated = output_logits[:, :min_len, :]
        target_programs_truncated = target_programs[:, :min_len]
        
        # Create mask for valid positions
        mask = torch.arange(min_len, device=program_lengths.device).unsqueeze(0) < program_lengths.unsqueeze(1)
        
        # Compute cross-entropy loss only on valid positions
        if mask.sum() > 0:
            valid_logits = output_logits_truncated.reshape(-1, vocab_size)[mask.flatten()]
            valid_targets = target_programs_truncated.flatten()[mask.flatten()]
            loss = F.cross_entropy(valid_logits, valid_targets, reduction='mean')
        else:
            loss = torch.tensor(0.0, device=output_logits.device)
        
        return loss
    
    def _compute_latent_behavior_loss(
        self,
        program_embeddings: torch.Tensor,
        states: torch.Tensor,
        target_actions: torch.Tensor,
        action_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute latent behavior reconstruction loss (L^L from paper)
        
        This ensures that programs with similar execution behaviors
        are close in the latent space.
        
        Args:
            program_embeddings: [batch_size, latent_dim] - compressed program embeddings
            states: [batch_size, num_demos, demo_len, H, W, C] - execution states
            target_actions: [batch_size, num_demos, demo_len-1] - target actions
            action_lengths: [batch_size, num_demos] - actual action lengths
            
        Returns:
            Latent behavior reconstruction loss
        """
        batch_size, num_demos = states.shape[:2]
        
        # Flatten states to [batch_size * num_demos, state_dim]
        states_flat = states[:, :, 0].reshape(batch_size * num_demos, -1)  # Use initial states
        
        # Expand program embeddings for each demo
        embeddings_expanded = program_embeddings.unsqueeze(1).expand(-1, num_demos, -1)
        embeddings_flat = embeddings_expanded.reshape(batch_size * num_demos, -1)
        
        # Flatten target actions and lengths
        target_actions_flat = target_actions.reshape(batch_size * num_demos, -1)
        action_lengths_flat = action_lengths.reshape(batch_size * num_demos)
        
        # Predict actions using neural executor
        predicted_action_logits = self.neural_executor(
            states_flat, 
            embeddings_flat,
            max_length=target_actions_flat.size(1)
        )
        
        # Create mask for valid positions
        max_action_len = target_actions_flat.size(1)
        mask = torch.arange(max_action_len, device=action_lengths_flat.device).unsqueeze(0) < action_lengths_flat.unsqueeze(1)
        
        # Compute cross-entropy loss
        if mask.sum() > 0:
            loss = F.cross_entropy(
                predicted_action_logits.reshape(-1, predicted_action_logits.size(-1)),
                target_actions_flat.reshape(-1),
                reduction='none'
            )
            masked_loss = loss.reshape(batch_size * num_demos, max_action_len) * mask.float()
            total_loss = masked_loss.sum() / mask.sum().float()
        else:
            total_loss = torch.tensor(0.0, device=program_embeddings.device)
        
        return total_loss
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.vae_uncompressed.train()
        self.compression_encoder.train()
        self.compression_decoder.train()
        self.neural_executor.train()
        
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Unpack batch
            programs, program_ids, masks, states, target_actions, action_lengths = batch
            
            # Move to device and ensure correct data types
            programs = programs.to(self.device).long()
            states = states.to(self.device).float()
            target_actions = target_actions.to(self.device).long()  # Ensure long type for cross_entropy
            action_lengths = action_lengths.to(self.device).long()
            
            # Get actual program lengths from masks
            program_lengths = masks.sum(dim=1).squeeze(-1).to(self.device).long()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            results = self.forward_pass(
                programs, program_lengths, states, target_actions, action_lengths
            )
            
            # Backward pass
            results['total_loss'].backward()
            
            # Gradient clipping
            all_params = []
            all_params.extend(self.vae_uncompressed.parameters())
            all_params.extend(self.compression_encoder.parameters())
            all_params.extend(self.compression_decoder.parameters())
            all_params.extend(self.neural_executor.parameters())
            
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            
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
                'behavior': f'{results["behavior_loss"].item():.4f}'
            })
            
            # Log to wandb
            if self.use_wandb and WANDB_AVAILABLE and self.global_step % 100 == 0:
                wandb.log({
                    f'train/{key}': value.item() if isinstance(value, torch.Tensor) else value
                    for key, value in results.items() if 'loss' in key
                }, step=self.global_step)
        
        # Average metrics
        if num_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
        
        return dict(epoch_metrics)
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.vae_uncompressed.eval()
        self.compression_encoder.eval()
        self.compression_decoder.eval()
        self.neural_executor.eval()
        
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Unpack batch
                programs, program_ids, masks, states, target_actions, action_lengths = batch
                
                # Move to device and ensure correct data types
                programs = programs.to(self.device).long()
                states = states.to(self.device).float()
                target_actions = target_actions.to(self.device).long()  # Ensure long type
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
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'vae_uncompressed_state_dict': self.vae_uncompressed.state_dict(),
            'compression_encoder_state_dict': self.compression_encoder.state_dict(),
            'compression_decoder_state_dict': self.compression_decoder.state_dict(),
            'neural_executor_state_dict': self.neural_executor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics)
        }
        
        '''# Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)'''
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch}")
        
        # Save latest model
        latest_path = os.path.join(self.checkpoint_dir, 'latest_model.pt')
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.vae_uncompressed.load_state_dict(checkpoint['vae_uncompressed_state_dict'])
        self.compression_encoder.load_state_dict(checkpoint['compression_encoder_state_dict'])
        self.compression_decoder.load_state_dict(checkpoint['compression_decoder_state_dict'])
        self.neural_executor.load_state_dict(checkpoint['neural_executor_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, resume_from: Optional[str] = None):
        """
        Main training loop following Algorithm 1 from the paper
        
        Args:
            resume_from: Path to checkpoint to resume from
        """
        if resume_from:
            self.load_checkpoint(resume_from)
        
        self.logger.info("Starting VAE training...")
        self.logger.info(f"Training for {self.num_epochs} epochs")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Beta (KL weight): {self.beta}")
        self.logger.info(f"Lambda (behavior weight): {self.lambda_behavior}")
        
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
                
                # Log to wandb
                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        **{f'train_epoch/{k}': v for k, v in train_metrics.items()},
                        **{f'val_epoch/{k}': v for k, v in val_metrics.items()},
                        'epoch': epoch,
                        'learning_rate': self.scheduler.get_last_lr()[0]
                    }, step=epoch)
                
                # Save checkpoint
                val_loss = val_metrics.get('total_loss', float('inf'))
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                if epoch % self.config.get('save_interval', 10) == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)
                
                '''# Early stopping check
                if self._should_early_stop():
                    self.logger.info("Early stopping triggered")
                    break'''
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        self.logger.info("Training completed")
        
        # Save final model
        self.save_checkpoint(self.current_epoch, False)
    
    def _should_early_stop(self, patience: int = 20) -> bool:
        """Check if training should stop early"""
        if len(self.val_metrics['total_loss']) < patience:
            return False
        
        recent_losses = self.val_metrics['total_loss'][-patience:]
        return all(loss >= self.best_val_loss for loss in recent_losses)
    
    def evaluate(self, checkpoint_path: str = None):
        """Evaluate the trained model"""
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.logger.info("Evaluating model...")
        
        # Run validation
        val_metrics = self.validate_epoch()
        
        self.logger.info("Evaluation Results:")
        for key, value in val_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
        
        return val_metrics
    
    def generate_samples(self, num_samples: int = 10) -> List[str]:
        """Generate sample programs from the trained VAE"""
        self.vae_uncompressed.eval()
        self.compression_encoder.eval()
        self.compression_decoder.eval()
        
        generated_programs = []
        
        with torch.no_grad():
            # Sample from prior
            z_compressed = torch.randn(num_samples, self.latent_dim, device=self.device)
            
            # Decompress
            z_uncompressed = self.compression_decoder(z_compressed)
            
            # Decode to programs
            output_logits, _ = self.vae_uncompressed.decoder(z_uncompressed, deterministic=True)
            
            # Convert to programs
            predicted_tokens = output_logits.argmax(dim=-1)
            
            for i in range(num_samples):
                tokens = predicted_tokens[i].cpu().numpy()
                # Remove padding
                tokens = tokens[tokens != self.padding_idx]
                program_str = self.dsl.intseq2str(tokens.tolist())
                generated_programs.append(program_str)
        
        return generated_programs


# Example usage and testing
if __name__ == "__main__":
    # Handle NumPy compatibility
    import warnings
    warnings.filterwarnings("ignore", message=".*NumPy.*")
    
    # Setup configuration
    test_config = config.copy()
    test_config['train']['max_epoch'] = 2  # Short training for testing
    test_config['train']['batch_size'] = 4
    test_config['valid']['batch_size'] = 4
    
    # Create trainer with dummy data for fast testing
    trainer = HPRLVAETrainer(
        config=test_config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_wandb=False,
        checkpoint_dir='./test_checkpoints'
    )
    
    # Force use of dummy data for testing
    trainer._use_dummy_data = True
    trainer._setup_data_loaders()  # Re-setup with dummy data
    
    print("VAE Trainer created successfully!")
    
    # Test training for a few steps
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