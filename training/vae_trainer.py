"""
FIXED VAE Trainer for HPRL - Anti-Collapse Version

Key fixes to prevent posterior collapse:
1. Much higher beta coefficients (0.5 instead of 0.01)
2. Lower learning rate to prevent rapid collapse
3. KL loss monitoring and early stopping
4. Target KL loss > 0.1 to maintain meaningful latent space
5. Better beta annealing schedule
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

from models.vae import ProgramVAE
from models.program_executor import ProgramExecutor
from training.data_loader import make_datasets, ProgramDataset
from dsl.karel_dsl import get_DSL_option_v2
from dsl.tokens import get_vocab_size, get_padding_index
from environments.karel_env import KarelEnvironment
from utils.config import config


class HPRLVAETrainer:
    """
    FIXED VAE Trainer for HPRL - Prevents Posterior Collapse
    
    Critical changes:
    - Higher beta coefficients (0.5 instead of 0.01)
    - KL loss monitoring and early stopping
    - Target KL loss > 0.1
    - Lower learning rate for stability
    - Better annealing schedule
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = 'cuda',
        use_wandb: bool = False,
        checkpoint_dir: str = './checkpoints'
    ):
        """Initialize FIXED VAE Trainer with anti-collapse settings"""
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
        self.latent_dim = 64
        self.max_program_length = config.get('dsl', {}).get('max_program_len', 12) + 1
        self.max_demo_length = config.get('rl', {}).get('envs', {}).get('executable', {}).get('max_demo_length', 100)
        
        # FIXED: Anti-collapse training parameters
        self.base_beta = config.get('loss', {}).get('latent_loss_coef', 0.5)  # Much higher default
        self.base_lambda_behavior = config.get('loss', {}).get('condition_loss_coef', 0.0)
        self.num_epochs = config.get('train', {}).get('max_epoch', 150)  # More epochs
        self.batch_size = config.get('train', {}).get('batch_size', 128)
        self.learning_rate = config.get('optimizer', {}).get('params', {}).get('lr', 5e-5)  # Lower LR
        
        # CRITICAL: Target KL loss to prevent collapse
        self.target_kl_loss = 0.1  # Minimum KL loss to maintain
        self.kl_collapse_threshold = 0.01  # Stop training if KL drops below this
        
        # FIXED: Create anti-collapse schedules
        self.beta_schedule = self._create_beta_schedule()
        self.lambda_schedule = self._create_lambda_schedule()
        
        # Set config defaults
        self._setup_config_defaults()
        
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
        
        # ADDED: Collapse monitoring
        self.collapse_warnings = 0
        self.kl_collapse_detected = False
        self.consecutive_low_kl = 0
        
        self.logger.info("FIXED VAE Trainer initialized with anti-collapse settings")
        self.logger.info(f"Beta schedule: start={self.beta_schedule[0]:.4f}, end={self.beta_schedule[-1]:.4f}")
        self.logger.info(f"Target KL loss: > {self.target_kl_loss}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.generation_training_start = 3000  # When to start generation training
        self.generation_training_frequency = 3  # Train generation every N batches  
        self.generation_weight = 0.3  # Weight for pure generation loss
    def _create_beta_schedule(self) -> List[float]:
        """IMPROVED: More aggressive beta annealing for better generation without teacher forcing"""
        schedule = []
        for epoch in range(self.num_epochs):
            if epoch < 5:
                # Start higher to prevent immediate collapse without teacher forcing
                beta = 0.1
            elif epoch < 20:
                # Quick ramp to substantial KL weight
                progress = (epoch - 5) / 15
                beta = 0.1 + progress * 0.4  # 0.1 → 0.5
            elif epoch < 80:
                # Maintain strong KL weight for generation training
                beta = 0.5
            elif epoch < 150:
                # Increase for better generation quality
                beta = 0.6
            else:
                # Maximum for final training
                beta = min(0.8, self.base_beta)
            schedule.append(beta)
        return schedule
    
    def _create_lambda_schedule(self) -> List[float]:
        """Create lambda schedule to gradually add action loss"""
        schedule = []
        for epoch in range(self.num_epochs):
            if epoch < 80:
                # Focus on VAE first
                lambda_val = 1
            elif epoch < 120:
                # Gradually add action loss
                progress = (epoch - 80) / 40
                lambda_val = self.base_lambda_behavior * progress
            else:
                # Use full action loss
                lambda_val = self.base_lambda_behavior
            schedule.append(lambda_val)
        return schedule
    
    def _setup_config_defaults(self):
        """Setup config defaults for compatibility"""
        self.config['use_simplified_dsl'] = self.config.get('use_simplified_dsl', False)
        if 'dsl_tokens' not in self.config:
            self.config['dsl_tokens'] = self.dsl.int2token
        if 'prl_tokens' not in self.config:
            self.config['prl_tokens'] = self.dsl.int2token
        if 'dsl2prl_mapping' not in self.config:
            self.config['dsl2prl_mapping'] = {token: token for token in self.dsl.int2token}
        if 'max_demo_length' not in self.config:
            self.config['max_demo_length'] = self.max_demo_length
        if 'dsl' not in self.config:
            self.config['dsl'] = {}
        if 'num_agent_actions' not in self.config['dsl']:
            self.config['dsl']['num_agent_actions'] = len(self.dsl.action_functions) + 1
    
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
        self.logger = logging.getLogger('FixedVAETrainer')
    
    def _build_models(self):
        """Build ProgramVAE model"""
        self.logger.info("Building ProgramVAE model...")
        
        self.program_vae = ProgramVAE(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            latent_dim=self.latent_dim,
            state_shape=(8, 8, 8),
            num_actions=self.config['dsl']['num_agent_actions'],
            max_program_length=self.max_program_length,
            max_demo_length=self.max_demo_length,
            dropout=self.config['net']['dropout'],
            rnn_type=self.config['net']['rnn_type']
        ).to(self.device)
        
        self.program_executor = ProgramExecutor(
            vocab_size=self.vocab_size,
            max_program_length=self.max_program_length,
            max_execution_steps=100,
            device=self.device
        )
        
        self.logger.info(f"ProgramVAE built - Total params: {sum(p.numel() for p in self.program_vae.parameters()):,}")
    
    def _setup_optimizers(self):
        """Setup optimizer with lower learning rate"""
        self.optimizer = optim.Adam(
            self.program_vae.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-5  # Added weight decay for stability
        )
        
        # FIXED: More conservative scheduler
        scheduler_config = self.config.get('optimizer', {}).get('scheduler', {})
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.8,
            patience=10
        )
        
        self.logger.info(f"Optimizer setup with LR={self.learning_rate}")
    
    def _setup_data_loaders(self):
        """Setup data loaders"""
        self.logger.info("Setting up data loaders...")
        
        if hasattr(self, '_use_dummy_data') and self._use_dummy_data:
            self.logger.info("Using dummy data for testing...")
            self._create_dummy_datasets()
            return
        
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
            batch_size=self.config.get('valid', {}).get('batch_size', 128),
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
        
        num_samples = 100
        dummy_programs = torch.randint(0, self.vocab_size-1, (num_samples, self.max_program_length))
        dummy_program_ids = torch.arange(num_samples)
        dummy_lengths = torch.randint(5, self.max_program_length, (num_samples,))
        dummy_masks = torch.ones(num_samples, self.max_program_length, 1, dtype=torch.bool)
        
        for i, length in enumerate(dummy_lengths):
            dummy_masks[i, length:] = False
        
        num_demos = 3
        demo_length = 10
        dummy_states = torch.randn(num_samples, num_demos, demo_length, 8, 8, 8)
        dummy_actions = torch.randint(0, 5, (num_samples, num_demos, demo_length-1), dtype=torch.long)
        dummy_action_lengths = torch.randint(1, demo_length-1, (num_samples, num_demos), dtype=torch.long)
        
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
        IMPROVED forward pass with dual training strategy - NO teacher forcing
        """
        # Get current coefficients from schedules
        current_beta = self.beta_schedule[min(self.current_epoch, len(self.beta_schedule)-1)]
        current_lambda = self.lambda_schedule[min(self.current_epoch, len(self.lambda_schedule)-1)]
        
        # Use action loss only when lambda > 0
        use_action_loss = current_lambda > 0
        
        # 1. RECONSTRUCTION TRAINING (encode real programs, decode without teacher forcing)
        mu, logvar = self.program_vae.vae.encode(programs, program_lengths)
        latent = self.program_vae.vae.reparameterize(mu, logvar)
        
        # Decode WITHOUT teacher forcing - pure generation from latent
        output_logits, output_log_probs = self.program_vae.vae.decode(
            latent, 
            target_programs=None,  # NO teacher forcing!
            deterministic=False
        )
        
        # Compute reconstruction loss against original programs
        recon_loss = self.compute_reconstruction_loss(output_logits, programs, program_lengths)
        kl_loss = self.program_vae.vae.kl_loss(mu, logvar)
        
        total_loss = recon_loss + current_beta * kl_loss
        
        losses = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'mu': mu,
            'logvar': logvar
        }
        
        # 2. PURE GENERATION TRAINING (random latents, no teacher forcing)
        if self.global_step > self.generation_training_start and self.global_step % 3 == 0:
            batch_size = programs.size(0)
            
            # Sample random latent vectors
            random_latents = torch.randn(batch_size, self.latent_dim, device=self.device)
            
            # Pure generation
            gen_logits, _ = self.program_vae.vae.decode(
                random_latents, 
                target_programs=None,  # NO teacher forcing!
                deterministic=False
            )
            
            # Simple coherence loss - encourage valid program structure
            coherence_loss = self.compute_coherence_loss(gen_logits)
            
            losses['total_loss'] += 0.3 * coherence_loss
            losses['generation_loss'] = coherence_loss
        else:
            losses['generation_loss'] = torch.tensor(0.0, device=self.device)
        
        # 3. ACTION PREDICTION (if enabled)
        if use_action_loss and states is not None:
            action_logits, action_log_probs = self.program_vae.condition_policy(
                states, latent, target_actions, deterministic=False
            )
            
            action_loss = self.compute_action_loss(action_logits, target_actions, action_lengths)
            losses['total_loss'] += current_lambda * action_loss
            losses['action_loss'] = action_loss
        else:
            losses['action_loss'] = torch.tensor(0.0, device=self.device)
        
        # Store training parameters
        losses['current_beta'] = torch.tensor(current_beta)
        losses['current_lambda'] = torch.tensor(current_lambda)
        
        # Monitor for collapse
        self._monitor_and_enforce_kl(losses, current_beta, current_lambda)
        
        return losses

    def compute_reconstruction_loss(self, output_logits: torch.Tensor, target_programs: torch.Tensor, program_lengths: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss without teacher forcing"""
        batch_size, seq_len, vocab_size = output_logits.shape
        target_seq_len = target_programs.size(1)
        
        # Match sequence lengths
        min_len = min(seq_len, target_seq_len)
        output_logits_matched = output_logits[:, :min_len, :]
        target_programs_matched = target_programs[:, :min_len]
        
        # Create mask for valid positions
        mask = torch.arange(min_len, device=program_lengths.device).unsqueeze(0) < program_lengths.unsqueeze(1)
        
        # Compute loss only on valid positions
        valid_positions = mask.flatten()
        if valid_positions.sum() > 0:
            valid_logits = output_logits_matched.reshape(-1, vocab_size)[valid_positions]
            valid_targets = target_programs_matched.flatten()[valid_positions]
            return F.cross_entropy(valid_logits, valid_targets, reduction='mean')
        else:
            return torch.tensor(0.0, device=output_logits.device, requires_grad=True)

    def compute_coherence_loss(self, gen_logits: torch.Tensor) -> torch.Tensor:
        """Compute coherence loss for pure generation to encourage valid program structure"""
        batch_size, seq_len, vocab_size = gen_logits.shape
        
        # Convert logits to tokens
        generated_tokens = torch.argmax(gen_logits, dim=-1)
        
        # Simple coherence targets - encourage SOS at start, reasonable structure
        coherence_targets = torch.zeros_like(generated_tokens)
        
        # First token should be SOS (token 0)
        coherence_targets[:, 0] = 0
        
        # Encourage early EOS for simple programs (token 1)
        if seq_len > 1:
            coherence_targets[:, 1] = 1
        
        # Rest should be padding (self.padding_idx)
        if seq_len > 2:
            coherence_targets[:, 2:] = self.padding_idx
        
        # Compute coherence loss
        coherence_loss = F.cross_entropy(
            gen_logits.reshape(-1, vocab_size),
            coherence_targets.reshape(-1),
            ignore_index=self.padding_idx,
            reduction='mean'
        )
        
        return coherence_loss

    def compute_action_loss(self, action_logits: torch.Tensor, target_actions: torch.Tensor, action_lengths: torch.Tensor) -> torch.Tensor:
        """Compute action prediction loss"""
        # Transpose action_logits to get [batch_size, num_demos, seq_len-1, num_actions]
        action_logits_transposed = action_logits.transpose(-2, -1)
        
        # Match sequence lengths
        actual_seq_len = action_logits_transposed.size(2)
        target_seq_len = target_actions.size(2)
        min_len = min(actual_seq_len, target_seq_len)
        
        action_logits_matched = action_logits_transposed[:, :, :min_len, :]
        target_actions_matched = target_actions[:, :, :min_len]
        
        # Flatten for cross entropy
        action_logits_flat = action_logits_matched.reshape(-1, action_logits_matched.size(-1))
        target_actions_flat = target_actions_matched.reshape(-1)
        
        return F.cross_entropy(action_logits_flat, target_actions_flat, reduction='mean')
    
    def _monitor_and_enforce_kl(self, losses: Dict[str, torch.Tensor], beta: float, lambda_val: float):
        """CRITICAL: Monitor and enforce minimum KL loss to prevent collapse"""
        recon_loss = losses['recon_loss'].item()
        kl_loss = losses['kl_loss'].item()
        
        # Check for collapse
        if kl_loss < self.kl_collapse_threshold and beta > 0.1:
            self.consecutive_low_kl += 1
            self.logger.error(f"🚨 KL COLLAPSE WARNING {self.consecutive_low_kl}/5: KL={kl_loss:.6f} < {self.kl_collapse_threshold}")
            
            if self.consecutive_low_kl >= 5:
                self.logger.error("🚨 STOPPING TRAINING: KL collapse detected!")
                self.kl_collapse_detected = True
                raise ValueError("Training stopped due to KL collapse")
        else:
            self.consecutive_low_kl = 0
        
        # Monitor KL health
        if kl_loss < self.target_kl_loss and beta > 0.1:
            self.logger.warning(f"⚠️ Low KL: {kl_loss:.4f} (target: >{self.target_kl_loss})")
        elif kl_loss >= self.target_kl_loss:
            if self.global_step % 100 == 0:
                self.logger.info(f"✅ Healthy KL: {kl_loss:.4f} (target: >{self.target_kl_loss})")
        
        # Detailed logging every 100 steps
        if self.global_step % 100 == 0:
            self.logger.info(f"Step {self.global_step}: recon={recon_loss:.4f}, kl={kl_loss:.4f}, "
                           f"β={beta:.4f}, λ={lambda_val:.4f}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with collapse monitoring"""
        self.program_vae.train()
        
        current_beta = self.beta_schedule[min(self.current_epoch, len(self.beta_schedule)-1)]
        current_lambda = self.lambda_schedule[min(self.current_epoch, len(self.lambda_schedule)-1)]
        
        self.logger.info(f"Epoch {self.current_epoch}: β={current_beta:.4f}, λ={current_lambda:.4f}")
        
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            programs, program_ids, masks, states, target_actions, action_lengths = batch
            
            programs = programs.to(self.device).long()
            states = states.to(self.device).float()
            target_actions = target_actions.to(self.device).long()
            action_lengths = action_lengths.to(self.device).long()
            
            program_lengths = masks.sum(dim=1).squeeze(-1).to(self.device).long()
            
            self.optimizer.zero_grad()
            
            try:
                # Forward pass with collapse monitoring
                results = self.forward_pass(
                    programs, program_lengths, states, target_actions, action_lengths
                )
                
                # Backward pass
                results['total_loss'].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.program_vae.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Update metrics
                for key, value in results.items():
                    if isinstance(value, torch.Tensor) and 'loss' in key:
                        epoch_metrics[key] += value.item()
                
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar
                current_loss = results['total_loss'].item()
                current_kl = results['kl_loss'].item()
                progress_bar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'recon': f'{results["recon_loss"].item():.4f}',
                    'kl': f'{current_kl:.4f}',
                    'β': f'{current_beta:.3f}',
                    'target_kl': f'>{self.target_kl_loss}'
                })
                
                # Check for collapse during training
                if self.kl_collapse_detected:
                    self.logger.error("Training stopped due to KL collapse!")
                    break
                    
            except ValueError as e:
                if "KL collapse" in str(e):
                    self.logger.error("Training terminated due to KL collapse")
                    break
                else:
                    raise e
        
        # Average metrics
        if num_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
        
        return dict(epoch_metrics)
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch with KL monitoring"""
        self.program_vae.eval()
        
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                programs, program_ids, masks, states, target_actions, action_lengths = batch
                
                programs = programs.to(self.device).long()
                states = states.to(self.device).float()
                target_actions = target_actions.to(self.device).long()
                action_lengths = action_lengths.to(self.device).long()
                
                program_lengths = masks.sum(dim=1).squeeze(-1).to(self.device).long()
                
                results = self.forward_pass(
                    programs, program_lengths, states, target_actions, action_lengths
                )
                
                for key, value in results.items():
                    if isinstance(value, torch.Tensor) and 'loss' in key:
                        epoch_metrics[key] += value.item()
                
                num_batches += 1
        
        if num_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
        
        # Check validation KL health
        val_kl = epoch_metrics.get('kl_loss', 0)
        current_beta = self.beta_schedule[min(self.current_epoch, len(self.beta_schedule)-1)]
        
        if val_kl < self.target_kl_loss and current_beta > 0.1:
            self.logger.warning(f"⚠️ Validation KL low: {val_kl:.4f} (target: >{self.target_kl_loss})")
        
        return dict(epoch_metrics)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint with anti-collapse info"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'program_vae_state_dict': self.program_vae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics),
            'beta_schedule': self.beta_schedule,
            'lambda_schedule': self.lambda_schedule,
            'collapse_warnings': self.collapse_warnings,
            'target_kl_loss': self.target_kl_loss,
            'kl_collapse_detected': self.kl_collapse_detected
        }
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch}")
        
        latest_path = os.path.join(self.checkpoint_dir, 'latest_model.pt')
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.program_vae.load_state_dict(checkpoint['program_vae_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') and hasattr(self.scheduler, 'load_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Load schedules and collapse info if available
        if 'beta_schedule' in checkpoint:
            self.beta_schedule = checkpoint['beta_schedule']
        if 'lambda_schedule' in checkpoint:
            self.lambda_schedule = checkpoint['lambda_schedule']
        if 'collapse_warnings' in checkpoint:
            self.collapse_warnings = checkpoint['collapse_warnings']
        if 'kl_collapse_detected' in checkpoint:
            self.kl_collapse_detected = checkpoint['kl_collapse_detected']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop with anti-collapse monitoring"""
        if resume_from:
            self.load_checkpoint(resume_from)
        
        self.logger.info("Starting FIXED VAE training with anti-collapse protection...")
        self.logger.info(f"Training for {self.num_epochs} epochs")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Beta schedule: {self.beta_schedule[0]:.4f} → {self.beta_schedule[-1]:.4f}")
        self.logger.info(f"Target KL loss: > {self.target_kl_loss}")
        self.logger.info(f"KL collapse threshold: < {self.kl_collapse_threshold}")
        
        try:
            for epoch in range(self.current_epoch, self.num_epochs):
                self.current_epoch = epoch
                
                # Check if collapse was detected
                if self.kl_collapse_detected:
                    self.logger.error("Training stopped due to KL collapse detection")
                    break
                
                # Training
                train_metrics = self.train_epoch()
                
                if self.kl_collapse_detected:
                    break
                
                # Validation
                val_metrics = self.validate_epoch()
                
                # Update learning rate scheduler
                val_loss = val_metrics.get('total_loss', float('inf'))
                if hasattr(self.scheduler, 'step') and hasattr(self.scheduler, 'mode'):
                    self.scheduler.step(val_loss)
                
                # Log epoch results
                train_kl = train_metrics.get('kl_loss', 0)
                val_kl = val_metrics.get('kl_loss', 0)
                
                self.logger.info(f"Epoch {epoch} - Train Loss: {train_metrics.get('total_loss', 0):.4f}, "
                               f"Val Loss: {val_metrics.get('total_loss', 0):.4f}")
                self.logger.info(f"  KL Losses - Train: {train_kl:.4f}, Val: {val_kl:.4f} (target: >{self.target_kl_loss})")
                
                # Health check
                if train_kl > self.target_kl_loss and val_kl > self.target_kl_loss:
                    self.logger.info(f"✅ Healthy training: KL losses above target")
                elif epoch > 10:  # Allow some initial epochs for warmup
                    self.logger.warning(f"⚠️ KL health concern: train={train_kl:.4f}, val={val_kl:.4f}")
                
                # Save metrics
                for key, value in train_metrics.items():
                    self.train_metrics[key].append(value)
                for key, value in val_metrics.items():
                    self.val_metrics[key].append(value)
                
                # Save checkpoint
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                if epoch % 5 == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        self.logger.info("Training completed")
        self.save_checkpoint(self.current_epoch, False)
        
        # Final health report
        final_train_kl = self.train_metrics.get('kl_loss', [0])[-1] if self.train_metrics.get('kl_loss') else 0
        final_val_kl = self.val_metrics.get('kl_loss', [0])[-1] if self.val_metrics.get('kl_loss') else 0
        
        self.logger.info(f"\n🎯 FINAL TRAINING REPORT:")
        self.logger.info(f"  Final KL losses - Train: {final_train_kl:.4f}, Val: {final_val_kl:.4f}")
        self.logger.info(f"  Target KL loss: > {self.target_kl_loss}")
        
        if final_train_kl > self.target_kl_loss and final_val_kl > self.target_kl_loss:
            self.logger.info(f"  ✅ SUCCESS: Maintained healthy KL losses!")
            self.logger.info(f"  🎉 VAE should have meaningful latent space for HPRL")
        elif self.kl_collapse_detected:
            self.logger.error(f"  ❌ FAILURE: KL collapse detected during training")
            self.logger.error(f"  💡 Try: Higher beta, lower learning rate, or different architecture")
        else:
            self.logger.warning(f"  ⚠️ PARTIAL: KL losses below target but no collapse detected")
            self.logger.warning(f"  💡 Consider: Higher beta coefficient for next training")
    
    def generate_samples(self, num_samples: int = 10) -> List[str]:
        """Generate sample programs from the trained model"""
        self.program_vae.eval()
        
        generated_programs = []
        
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            
            # Generate programs (this may still fail if collapse occurred)
            try:
                predicted_tokens = self.program_vae.generate_program(z, deterministic=True)
                
                for i in range(num_samples):
                    tokens = predicted_tokens[i].cpu().numpy()
                    tokens = tokens[tokens != self.padding_idx]
                    program_str = self.dsl.intseq2str(tokens.tolist())
                    generated_programs.append(program_str)
            except Exception as e:
                self.logger.warning(f"Generation failed: {e}")
                generated_programs = [f"<GENERATION_ERROR>"] * num_samples
        
        return generated_programs
    
    def test_latent_space_quality(self) -> Dict[str, Any]:
        """Test the quality of the learned latent space"""
        self.logger.info("Testing latent space quality...")
        
        test_programs = [
            "DEF run m( move m)",
            "DEF run m( turnLeft m)",
            "DEF run m( turnRight m)",
            "DEF run m( pickMarker m)",
            "DEF run m( putMarker m)"
        ]
        
        embeddings = []
        
        with torch.no_grad():
            for program in test_programs:
                try:
                    tokens = self.dsl.str2intseq(program)
                    padded_tokens = tokens + [self.padding_idx] * (self.max_program_length - len(tokens))
                    padded_tokens = padded_tokens[:self.max_program_length]
                    
                    tokens_tensor = torch.tensor([padded_tokens], dtype=torch.long, device=self.device)
                    lengths_tensor = torch.tensor([len(tokens)], device=self.device)
                    
                    mu, logvar = self.program_vae.vae.encode(tokens_tensor, lengths_tensor)
                    embeddings.append(mu.cpu().numpy()[0])
                except Exception as e:
                    self.logger.warning(f"Error encoding '{program}': {e}")
        
        if len(embeddings) < 2:
            return {'error': 'Could not encode enough programs for analysis'}
        
        embeddings = np.array(embeddings)
        
        # Calculate diversity metrics
        norms = np.linalg.norm(embeddings, axis=1)
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                distances.append(np.linalg.norm(embeddings[i] - embeddings[j]))
        
        mean_distance = np.mean(distances)
        dimension_stds = np.std(embeddings, axis=0)
        active_dimensions = np.sum(dimension_stds > 0.01)
        
        quality_report = {
            'num_programs_tested': len(test_programs),
            'mean_embedding_norm': float(np.mean(norms)),
            'mean_pairwise_distance': float(mean_distance),
            'active_dimensions': int(active_dimensions),
            'dimension_usage_ratio': float(active_dimensions / self.latent_dim),
            'embeddings_shape': embeddings.shape
        }
        
        # Assessment
        if mean_distance > 0.5 and active_dimensions > 10:
            quality_report['assessment'] = 'good'
            self.logger.info("✅ Latent space appears healthy")
        elif mean_distance > 0.1 and active_dimensions > 5:
            quality_report['assessment'] = 'moderate'
            self.logger.warning("⚠️ Latent space has moderate quality")
        else:
            quality_report['assessment'] = 'poor'
            self.logger.warning("❌ Latent space may be collapsed")
        
        self.logger.info(f"  Active dimensions: {active_dimensions}/{self.latent_dim}")
        self.logger.info(f"  Mean pairwise distance: {mean_distance:.4f}")
        
        return quality_report


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message=".*NumPy.*")
    
    # Test configuration with anti-collapse settings
    test_config = config.copy()
    test_config['train']['max_epoch'] = 10  # Short test
    test_config['train']['batch_size'] = 4
    test_config['valid']['batch_size'] = 4
    test_config['loss']['latent_loss_coef'] = 0.5  # Higher beta
    test_config['optimizer']['params']['lr'] = 5e-5  # Lower LR
    
    # Create trainer
    trainer = HPRLVAETrainer(
        config=test_config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_wandb=False,
        checkpoint_dir='./test_checkpoints_fixed'
    )
    
    # Force dummy data for testing
    trainer._use_dummy_data = True
    trainer._setup_data_loaders()
    
    print("✅ FIXED VAE Trainer created successfully!")
    print(f"🛡️ Anti-collapse settings:")
    print(f"  Beta schedule: {trainer.beta_schedule[:5]}... → {trainer.beta_schedule[-1]}")
    print(f"  Target KL loss: > {trainer.target_kl_loss}")
    print(f"  Learning rate: {trainer.learning_rate}")
    
    try:
        # Short training test
        trainer.train()
        print("✅ Training test completed!")
        
        # Test latent space quality
        quality_report = trainer.test_latent_space_quality()
        print(f"📊 Latent space assessment: {quality_report.get('assessment', 'unknown')}")
        
        # Test generation
        samples = trainer.generate_samples(num_samples=3)
        print("\n🎲 Generated samples:")
        for i, sample in enumerate(samples):
            print(f"  {i+1}: {sample}")
            
    except Exception as e:
        print(f"❌ Error during training test: {e}")
        import traceback
        traceback.print_exc()