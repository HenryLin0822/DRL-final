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
        #self.max_program_length = config.get('dsl', {}).get('max_program_len', 24) + 1
        self.max_program_length = 25
        print("max program length:",self.max_program_length)
        self.max_demo_length = config.get('rl', {}).get('envs', {}).get('executable', {}).get('max_demo_length', 100)
        
        # FIXED: Anti-collapse training parameters
        self.base_beta = config.get('loss', {}).get('latent_loss_coef', 0.5)  # Much higher default
        self.base_lambda_behavior = config.get('loss', {}).get('condition_loss_coef', 0.0)
        self.num_epochs = config.get('train', {}).get('max_epoch', 150)  # More epochs
        self.batch_size = config.get('train', {}).get('batch_size', 128)
        self.learning_rate = config.get('optimizer', {}).get('params', {}).get('lr', 5e-5)  # Lower LR
        
        # ADDED: Periodic saving configuration
        self.save_frequency = config.get('train', {}).get('save_frequency', 5)  # Save every N epochs
        self.save_on_improvement = config.get('train', {}).get('save_on_improvement', True)  # Save when validation improves
        self.keep_n_checkpoints = config.get('train', {}).get('keep_n_checkpoints', 5)  # Keep last N checkpoints
        
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
        
        # ADDED: Checkpoint tracking
        self.saved_checkpoints = []  # List to track saved checkpoint files
        
        self.logger.info("FIXED VAE Trainer initialized with anti-collapse settings")
        self.logger.info(f"Beta schedule: start={self.beta_schedule[0]:.4f}, end={self.beta_schedule[-1]:.4f}")
        self.logger.info(f"Target KL loss: > {self.target_kl_loss}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Periodic save: every {self.save_frequency} epochs, keep {self.keep_n_checkpoints} checkpoints")
        
        self.generation_training_start = 3000  # When to start generation training
        self.generation_training_frequency = 3  # Train generation every N batches  
        self.generation_weight = 0.1  # Weight for pure generation loss
    def _setup_optimizers(self):
        """Setup optimizer with cosine annealing scheduler"""
        self.optimizer = optim.Adam(
            self.program_vae.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-5  # Added weight decay for stability
        )
        
        # IMPROVED: Use Cosine Annealing as default
        scheduler_config = self.config.get('optimizer', {}).get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')  # Default to cosine
        
        if scheduler_type == 'cosine':
            # Cosine Annealing - smooth decay over training
            #T_max = scheduler_config.get('T_max', self.num_epochs)
            eta_min = scheduler_config.get('eta_min', self.learning_rate * 0.01)  # 1% of initial LR
            T_max= 20
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=eta_min
            )
            self.scheduler_type = 'cosine'
            self.logger.info(f"Cosine Annealing scheduler: T_max={T_max}, eta_min={eta_min:.2e}")
            
        elif scheduler_type == 'cosine_restarts':
            # Cosine Annealing with Warm Restarts
            T_0 = scheduler_config.get('T_0', 50)
            T_mult = scheduler_config.get('T_mult', 1)
            eta_min = scheduler_config.get('eta_min', self.learning_rate * 0.01)
            
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min
            )
            self.scheduler_type = 'cosine_restarts'
            self.logger.info(f"Cosine Annealing with Restarts: T_0={T_0}, T_mult={T_mult}, eta_min={eta_min:.2e}")
            
        elif scheduler_type == 'plateau':
            # Fallback to ReduceLROnPlateau
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                threshold=0.001,
                min_lr=1e-7,
                verbose=True
            )
            self.scheduler_type = 'plateau'
            self.logger.info(f"ReduceLROnPlateau scheduler: factor=0.5, patience=5")
            
        elif scheduler_type == 'step':
            # Step scheduler
            step_size = scheduler_config.get('step_size', 20)
            gamma = scheduler_config.get('gamma', 0.5)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
            self.scheduler_type = 'step'
            self.logger.info(f"StepLR scheduler: step_size={step_size}, gamma={gamma}")
            
        else:
            # No scheduler
            self.scheduler = None
            self.scheduler_type = 'none'
            self.logger.info("No learning rate scheduler")
        
        self.logger.info(f"Optimizer setup with initial LR={self.learning_rate}")
        self.logger.info(f"Scheduler type: {self.scheduler_type}")
    def _create_beta_schedule(self) -> List[float]:
        """IMPROVED: More aggressive beta annealing for better generation without teacher forcing"""
        schedule = []
        for epoch in range(self.num_epochs):
            if epoch < 30:
                # Start higher to prevent immediate collapse without teacher forcing
                beta = 0.1
            elif epoch < 50:
                # Quick ramp to substantial KL weight
                progress = (epoch - 50) / 20
                beta = 0.1 + progress * 0.4  # 0.1 → 0.5
            elif epoch < 80:
                # Maintain strong KL weight for generation training
                beta = 0.5
            elif epoch < 100:
                # Increase for better generation quality
                beta = 0.6
            else:
                # Maximum for final training
                beta = min(0.8, self.base_beta)
            schedule.append(beta*0.07)
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
    
    def forward_pass(
        self, 
        programs: torch.Tensor, 
        program_lengths: torch.Tensor,
        states: torch.Tensor,
        target_actions: torch.Tensor,
        action_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        ENHANCED forward pass with token-weighted losses
        """
        # Get current coefficients from schedules
        current_beta = self.beta_schedule[min(self.current_epoch, len(self.beta_schedule)-1)]
        current_lambda = self.lambda_schedule[min(self.current_epoch, len(self.lambda_schedule)-1)]
        
        # Use action loss only when lambda > 0
        use_action_loss = current_lambda > 0
        
        # 1. RECONSTRUCTION TRAINING with token weighting
        mu, logvar = self.program_vae.vae.encode(programs, program_lengths)
        latent = self.program_vae.vae.reparameterize(mu, logvar)
        
        # Decode WITHOUT teacher forcing
        output_logits, output_log_probs = self.program_vae.vae.decode(
            latent, 
            target_programs=None,
            deterministic=False
        )
        
        # TOKEN-WEIGHTED reconstruction loss
        recon_loss = self.compute_token_weighted_reconstruction_loss(output_logits, programs, program_lengths)
        kl_loss = self.program_vae.vae.kl_loss(mu, logvar)
        
        total_loss = recon_loss + current_beta * kl_loss
        
        losses = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'mu': mu,
            'logvar': logvar
        }
        
        # 2. ENHANCED GENERATION TRAINING with token weighting
        if self.global_step > self.generation_training_start and self.global_step % self.generation_training_frequency == 0:
            batch_size = programs.size(0)
            
            # Sample random latent vectors
            random_latents = torch.randn(batch_size, self.latent_dim, device=self.device)
            
            # Pure generation
            gen_logits, _ = self.program_vae.vae.decode(
                random_latents, 
                target_programs=None,
                deterministic=False
            )
            
            # TOKEN-WEIGHTED coherence loss
            coherence_loss = self.compute_token_weighted_coherence_loss(gen_logits, programs)
            
            losses['total_loss'] += self.generation_weight * coherence_loss
            losses['generation_loss'] = coherence_loss
        else:
            losses['generation_loss'] = torch.tensor(0.0, device=self.device)
        
        # 3. ACTION PREDICTION (unchanged)
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

    def compute_token_weighted_coherence_loss(self, gen_logits: torch.Tensor, reference_programs: torch.Tensor = None) -> torch.Tensor:
        """
        ENHANCED: Coherence loss that encourages generation of important tokens
        """
        batch_size, seq_len, vocab_size = gen_logits.shape
        device = gen_logits.device
        
        # Standard diversity loss
        probs = F.softmax(gen_logits, dim=-1)
        entropy_per_position = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        avg_entropy = entropy_per_position.mean()
        diversity_loss = -avg_entropy
        
        # ENCOURAGE IMPORTANT TOKEN GENERATION
        important_token_bonus = torch.tensor(0.0, device=device)
        
        '''# Define important tokens we want to encourage
        encourage_tokens = {
            'REPEAT': 0.1,
            'WHILE': 0.1, 
            'IF': 0.05,
            'IFELSE': 0.05,
            'frontIsClear': 0.03,
            'markersPresent': 0.03,
        }'''
        encourage_tokens = {
            'REPEAT': 0.03,

        }
        
        try:
            # Encourage generation of important tokens
            for token_str, bonus_weight in encourage_tokens.items():
                token_id = self.dsl.token2int.get(token_str, None)
                if token_id is not None and token_id < vocab_size:
                    # Sum probability of this token across all positions and batches
                    token_prob_sum = probs[:, :, token_id].sum()
                    
                    # Reward higher probability (negative loss)
                    important_token_bonus += bonus_weight * token_prob_sum
                    
        except Exception as e:
            pass  # Ignore if token mapping fails
        
        # PENALIZE NEVER GENERATING IMPORTANT TOKENS
        token_generation_penalty = torch.tensor(0.0, device=device)
        
        try:
            predicted_tokens = torch.argmax(gen_logits, dim=-1)  # [batch_size, seq_len]
            
            # Check if any important tokens were generated
            important_token_ids = []
            for token_str in ['REPEAT', 'WHILE', 'IF']:
                token_id = self.dsl.token2int.get(token_str, None)
                if token_id is not None:
                    important_token_ids.append(token_id)
            
            if important_token_ids:
                for token_id in important_token_ids:
                    # Check if this token appears in any generated sequence
                    token_appears = (predicted_tokens == token_id).any()
                    
                    if not token_appears:
                        # Small penalty for never generating important tokens
                        token_generation_penalty += 0.02
                        
        except Exception as e:
            pass
        
        # Combine losses
        total_loss = diversity_loss - important_token_bonus + token_generation_penalty
        
        return total_loss

    def _compute_data_driven_structure_loss(self, gen_probs: torch.Tensor, 
                                        reference_programs: torch.Tensor) -> torch.Tensor:
        """
        Helper function for data-driven structure loss (NEW HELPER)
        """
        batch_size, seq_len, vocab_size = gen_probs.shape
        ref_seq_len = min(seq_len, reference_programs.size(1))
        
        if ref_seq_len == 0:
            return torch.tensor(0.0, device=gen_probs.device)
        
        try:
            # Get token frequencies from reference programs (current batch)
            ref_tokens = reference_programs[:, :ref_seq_len]
            ref_mask = ref_tokens != self.padding_idx
            
            if ref_mask.sum() == 0:
                return torch.tensor(0.0, device=gen_probs.device)
            
            # Calculate empirical token distribution from real programs
            valid_ref_tokens = ref_tokens[ref_mask]
            ref_token_counts = torch.bincount(valid_ref_tokens, minlength=vocab_size)
            ref_token_probs = ref_token_counts.float() / (ref_token_counts.sum() + 1e-8)
            
            # Calculate generated token distribution
            gen_probs_flat = gen_probs[:, :ref_seq_len, :].reshape(-1, vocab_size)
            gen_token_probs = gen_probs_flat.mean(dim=0)
            
            # KL divergence: encourage generated distribution to match reference
            kl_div = F.kl_div(
                torch.log(gen_token_probs + 1e-8), 
                ref_token_probs + 1e-8, 
                reduction='sum'
            )
            
            return kl_div
            
        except Exception as e:
            return torch.tensor(0.0, device=gen_probs.device)

    def compute_token_weighted_reconstruction_loss(self, output_logits: torch.Tensor, target_programs: torch.Tensor, program_lengths: torch.Tensor) -> torch.Tensor:
        """
        ENHANCED: Reconstruction loss with higher weights for rare/important tokens
        """
        batch_size, seq_len, vocab_size = output_logits.shape
        target_seq_len = target_programs.size(1)
        min_len = min(seq_len, target_seq_len)
        
        if min_len == 0:
            return torch.tensor(0.0, device=output_logits.device, requires_grad=True)
        
        # Define token weights - higher weight for important/rare tokens
        token_weights = torch.ones(vocab_size, device=output_logits.device)
        
        # Define important tokens with higher weights
        '''  important_tokens = {
            'REPEAT': 5.0,    # 5x weight
            'WHILE': 5.0,     # 5x weight  
            'IF': 3.0,        # 3x weight
            'IFELSE': 3.0,    # 3x weight
            'frontIsClear': 2.0,
            'markersPresent': 2.0,
            'noMarkersPresent': 2.0,
            'R=2': 2.0, 'R=3': 2.0, 'R=4': 2.0, 'R=5': 2.0,
            'c(': 1.5, 'c)': 1.5,  # Condition brackets
            'w(': 1.5, 'w)': 1.5,  # While brackets
            'r(': 1.5, 'r)': 1.5,  # Repeat brackets
            'i(': 1.5, 'i)': 1.5,  # If brackets
            'e(': 1.5, 'e)': 1.5,  # Else brackets
        }'''
        important_tokens = {
            'REPEAT': 1.2,    # 5x weight
            'R=2': 1.2, 'R=3': 1.2, 'R=4': 1.2, 'R=5': 1.2,
            'r(': 1.2, 'r)': 1.2,  # Repeat brackets
        }
        # Map token strings to IDs and set weights
        try:
            for token_str, weight in important_tokens.items():
                token_id = self.dsl.token2int.get(token_str, None)
                if token_id is not None and token_id < vocab_size:
                    token_weights[token_id] = weight
                    '''if weight >= 3.0:  # Log high-weight tokens
                        print(f"High weight token: {token_str} (ID: {token_id}) = {weight}x")'''
        except Exception as e:
            print(f"Warning: Could not set token weights: {e}")
        
        total_loss = torch.tensor(0.0, device=output_logits.device, requires_grad=True)
        loss_count = 0
        
        for b in range(batch_size):
            actual_length = program_lengths[b].item()
            actual_length = min(actual_length, min_len)
            
            if actual_length <= 0:
                continue
            
            # 1. WEIGHTED reconstruction loss on actual program tokens
            program_logits = output_logits[b, :actual_length, :]  # [actual_length, vocab_size]
            program_targets = target_programs[b, :actual_length]   # [actual_length]
            
            if program_targets.numel() > 0:
                # Get weights for each target token
                target_token_weights = token_weights[program_targets]  # [actual_length]
                
                # Compute loss for each position
                position_losses = F.cross_entropy(
                    program_logits, program_targets, reduction='none'
                )  # [actual_length]
                
                # Apply token-specific weights
                weighted_losses = position_losses * target_token_weights
                
                # Average the weighted losses
                weighted_recon_loss = weighted_losses.mean()
                
                total_loss = total_loss + weighted_recon_loss
                loss_count += 1
            
            # 2. Keep termination loss (unchanged)
            if actual_length < min_len:
                termination_pos = actual_length
                
                if termination_pos < seq_len:
                    termination_logits = output_logits[b, termination_pos, :]
                    padding_loss = F.cross_entropy(
                        termination_logits.unsqueeze(0), 
                        torch.tensor([self.padding_idx], device=output_logits.device),
                        reduction='mean'
                    )
                    total_loss = total_loss + 2.0 * padding_loss
                    loss_count += 1
            
            # 3. Keep padding loss (unchanged)
            if actual_length < min_len - 1:
                padding_positions = slice(actual_length + 1, min_len)
                padding_logits = output_logits[b, padding_positions, :]
                
                if padding_logits.numel() > 0:
                    num_padding_positions = padding_logits.size(0)
                    padding_targets = torch.full(
                        (num_padding_positions,), 
                        self.padding_idx, 
                        device=output_logits.device
                    )
                    
                    padding_loss = F.cross_entropy(padding_logits, padding_targets, reduction='mean')
                    total_loss = total_loss + 3.0 * padding_loss
                    loss_count += 1
        
        return total_loss / loss_count if loss_count > 0 else total_loss
    

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
        """Train for one epoch with collapse monitoring and generation testing"""
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
        
        # ADDED: Test generation quality every epoch
        self.logger.info(f"\n🧪 Testing generation quality at end of epoch {self.current_epoch}...")
        self._test_generation_during_training()
        
        return dict(epoch_metrics)

    def _test_generation_during_training(self):
        """
        ENHANCED: Test generation with termination analysis (SAME FUNCTION NAME)
        """
        # Add tokenization debug (run once)
        if self.current_epoch == 0:
            self.debug_tokenization()
        
        self.program_vae.eval()
        
        with torch.no_grad():
            try:
                # 1. Test pure generation from random latents
                self.logger.info("   🎲 Testing pure generation...")
                test_latents = torch.randn(5, self.latent_dim, device=self.device)
                
                if hasattr(self.program_vae, 'generate_program_pure'):
                    try:
                        generated_tokens = self.program_vae.generate_program_pure(
                            test_latents, deterministic=True
                        )
                        for i, tokens in enumerate(generated_tokens):
                            program_str = self._detokenize_program(tokens)
                            # Calculate actual length (before padding)
                            actual_length = sum(1 for t in tokens if t.item() != self.padding_idx)
                            self.logger.info(f"      Generated {i+1}: {program_str} (len: {actual_length})")
                    except Exception as e:
                        self.logger.warning(f"      Pure generation failed: {e}")
                
                # 2. Test reconstruction of known programs
                self.logger.info("   🔄 Testing reconstruction...")
                test_programs = [
                    "DEF run m( move m)",
                    "DEF run m( turnLeft m)",
                    "DEF run m( pickMarker m)",
                    "DEF run m( IFELSE c( frontIsClear c) i( turnRight move move turnRight i) ELSE e( pickMarker move move turnLeft e) m)",
                    "DEF run m( turnLeft move pickMarker turnLeft m)",
                    "DEF run m( turnLeft move IF c( frontIsClear c) i( move i) putMarker WHILE c( frontIsClear c) w( move w) m)",
                ]
                
                for i, program_str in enumerate(test_programs):
                    try:
                        # Tokenize program
                        tokens = self.dsl.str2intseq(program_str)
                        padded_tokens = tokens + [self.padding_idx] * (self.max_program_length - len(tokens))
                        padded_tokens = padded_tokens[:self.max_program_length]
                        
                        tokens_tensor = torch.tensor([padded_tokens], dtype=torch.long, device=self.device)
                        lengths_tensor = torch.tensor([len(tokens)], device=self.device)
                        
                        # Encode and decode
                        mu, logvar = self.program_vae.vae.encode(tokens_tensor, lengths_tensor)
                        latent = self.program_vae.vae.reparameterize(mu, logvar)
                        
                        # Test no-teacher-forcing decode
                        output_logits, _ = self.program_vae.vae.decode(
                            latent, target_programs=None, deterministic=True
                        )
                        predicted_tokens = torch.argmax(output_logits, dim=-1)
                        reconstructed_str = self._detokenize_program(predicted_tokens[0])
                        
                        # Calculate actual lengths
                        original_length = len(tokens)
                        reconstructed_length = sum(1 for t in predicted_tokens[0] if t.item() != self.padding_idx)
                        
                        # Compute simple accuracy on original length
                        original_tokens = torch.tensor(tokens, device=self.device)
                        predicted_tokens_clean = predicted_tokens[0][:len(tokens)]
                        accuracy = (original_tokens == predicted_tokens_clean).float().mean().item()
                        
                        self.logger.info(f"      Original: {program_str} (len: {original_length})")
                        self.logger.info(f"      Reconstructed: {reconstructed_str} (len: {reconstructed_length}, acc: {accuracy:.2%})")
                        
                        # Check for over-generation
                        if reconstructed_length > original_length + 2:
                            self.logger.warning(f"        ⚠️ Over-generation: {reconstructed_length - original_length} extra tokens!")
                        
                    except Exception as e:
                        self.logger.warning(f"      Reconstruction failed for '{program_str}': {e}")
                
                # 3. Analyze sequence termination patterns
                self.logger.info("   📏 Analyzing sequence termination...")
                
                try:
                    test_latents = torch.randn(10, self.latent_dim, device=self.device)
                    
                    if hasattr(self.program_vae, 'generate_program_pure'):
                        generated_tokens = self.program_vae.generate_program_pure(
                            test_latents, deterministic=True
                        )
                        
                        lengths = []
                        natural_endings = 0
                        
                        for tokens in generated_tokens:
                            # Find actual length (before padding)
                            actual_length = 0
                            for token in tokens:
                                if token.item() == self.padding_idx:
                                    break
                                actual_length += 1
                            
                            lengths.append(actual_length)
                            
                            # Check if ended naturally (before max length)
                            if actual_length < self.max_program_length:
                                natural_endings += 1
                        
                        avg_length = sum(lengths) / len(lengths) if lengths else 0
                        self.logger.info(f"      Generated lengths: {lengths}")
                        self.logger.info(f"      Average length: {avg_length:.1f}")
                        self.logger.info(f"      Natural endings: {natural_endings}/{len(lengths)} ({natural_endings/len(lengths)*100:.1f}%)")
                        
                        if natural_endings < len(lengths) * 0.3:
                            self.logger.warning(f"      ⚠️ Few natural endings - sequences may not be learning to terminate!")
                        elif natural_endings > len(lengths) * 0.7:
                            self.logger.info(f"      ✅ Good termination behavior!")
                            
                except Exception as e:
                    self.logger.warning(f"      Termination analysis failed: {e}")
            
            except Exception as e:
                self.logger.error(f"   Generation testing failed: {e}")
        
        self.program_vae.train()


    def debug_tokenization(self):
        """
        NEW: Debug function to understand tokenization and EOS handling
        """
        test_programs = [
            "DEF run m( move m)",
            "DEF run m( turnLeft m)",
            "DEF run m( pickMarker m)"
        ]
        
        print("\n🔍 TOKENIZATION DEBUG:")
        print("=" * 50)
        
        for program in test_programs:
            print(f"\nProgram: '{program}'")
            
            try:
                tokens = self.dsl.str2intseq(program)
                print(f"Raw tokens: {tokens}")
                
                # Show token meanings
                token_meanings = []
                for token_id in tokens:
                    try:
                        token_str = self.dsl.int2token.get(token_id, f"<UNK_{token_id}>")
                        token_meanings.append(f"{token_id}:{token_str}")
                    except:
                        token_meanings.append(f"{token_id}:<error>")
                
                print(f"Token meanings: {token_meanings}")
                print(f"Program length: {len(tokens)}")
                
            except Exception as e:
                print(f"❌ Tokenization failed: {e}")
        
        print(f"\nPadding token: {self.padding_idx}")
        print(f"Max program length: {self.max_program_length}")
        print("=" * 50)

    def _detokenize_program(self, tokens: torch.Tensor) -> str:
        """Helper function to convert tokens back to program string"""
        if isinstance(tokens, torch.Tensor):
            tokens_list = tokens.cpu().numpy().tolist()
        else:
            tokens_list = tokens
        
        # Remove padding tokens
        cleaned_tokens = []
        for token in tokens_list:
            if token == self.padding_idx:
                break
            cleaned_tokens.append(token)
        
        if not cleaned_tokens:
            return "<EMPTY>"
        
        try:
            return self.dsl.intseq2str(cleaned_tokens)
        except Exception as e:
            return f"<PARSE_ERROR: {cleaned_tokens}>"
    
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
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, is_periodic: bool = False):
        """Save model checkpoint with periodic saving and cleanup"""
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
            'kl_collapse_detected': self.kl_collapse_detected,
            'save_timestamp': torch.utils.data.get_worker_info()  # For tracking
        }
        
        # Always save the latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_model.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 Saved best model at epoch {epoch}")
        
        # Periodic checkpoint saving
        if is_periodic or (epoch % self.save_frequency == 0):
            periodic_filename = f'checkpoint_epoch_{epoch:04d}.pt'
            periodic_path = os.path.join(self.checkpoint_dir, periodic_filename)
            torch.save(checkpoint, periodic_path)
            
            # Track saved checkpoints
            self.saved_checkpoints.append({
                'epoch': epoch,
                'filename': periodic_filename,
                'path': periodic_path,
                'val_loss': checkpoint.get('best_val_loss', float('inf'))
            })
            
            self.logger.info(f"💾 Saved periodic checkpoint: {periodic_filename}")
            
            # Cleanup old checkpoints (keep only last N)
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Remove old periodic checkpoints, keeping only the most recent N"""
        if len(self.saved_checkpoints) > self.keep_n_checkpoints:
            # Sort by epoch (oldest first)
            self.saved_checkpoints.sort(key=lambda x: x['epoch'])
            
            # Remove oldest checkpoints
            checkpoints_to_remove = self.saved_checkpoints[:-self.keep_n_checkpoints]
            
            for checkpoint_info in checkpoints_to_remove:
                try:
                    if os.path.exists(checkpoint_info['path']):
                        os.remove(checkpoint_info['path'])
                        self.logger.info(f"🗑️ Removed old checkpoint: {checkpoint_info['filename']}")
                except Exception as e:
                    self.logger.warning(f"Could not remove checkpoint {checkpoint_info['filename']}: {e}")
            
            # Update the list to keep only recent checkpoints
            self.saved_checkpoints = self.saved_checkpoints[-self.keep_n_checkpoints:]

    def list_saved_checkpoints(self):
        """List all available checkpoints"""
        print(f"\n📁 Available Checkpoints in {self.checkpoint_dir}:")
        
        # Check for special checkpoints
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        latest_path = os.path.join(self.checkpoint_dir, 'latest_model.pt')
        
        if os.path.exists(best_path):
            print(f"  🏆 best_model.pt")
        if os.path.exists(latest_path):
            print(f"  🔄 latest_model.pt")
        
        # List periodic checkpoints
        if self.saved_checkpoints:
            print(f"  📅 Periodic checkpoints:")
            for checkpoint_info in sorted(self.saved_checkpoints, key=lambda x: x['epoch'], reverse=True):
                print(f"    - {checkpoint_info['filename']} (epoch {checkpoint_info['epoch']}, val_loss: {checkpoint_info['val_loss']:.4f})")
        else:
            print(f"  📅 No periodic checkpoints yet")
        
        return self.saved_checkpoints
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.program_vae.load_state_dict(checkpoint['program_vae_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        """if checkpoint.get('scheduler_state_dict') and hasattr(self.scheduler, 'load_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])"""
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Load schedules and collapse info if available
        """if 'beta_schedule' in checkpoint:
            self.beta_schedule = checkpoint['beta_schedule']"""
        if 'lambda_schedule' in checkpoint:
            self.lambda_schedule = checkpoint['lambda_schedule']
        if 'collapse_warnings' in checkpoint:
            self.collapse_warnings = checkpoint['collapse_warnings']
        if 'kl_collapse_detected' in checkpoint:
            self.kl_collapse_detected = checkpoint['kl_collapse_detected']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop with anti-collapse monitoring and periodic saving"""
        if resume_from:
            self.load_checkpoint(resume_from)
        
        self.logger.info("Starting FIXED VAE training with anti-collapse protection...")
        self.logger.info(f"Training for {self.num_epochs} epochs")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Beta schedule: {self.beta_schedule[0]:.4f} → {self.beta_schedule[-1]:.4f}")
        self.logger.info(f"Target KL loss: > {self.target_kl_loss}")
        self.logger.info(f"KL collapse threshold: < {self.kl_collapse_threshold}")
        self.logger.info(f"💾 Periodic saving: every {self.save_frequency} epochs, keeping {self.keep_n_checkpoints} checkpoints")
        
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

                if self.scheduler is not None:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    
                    if self.scheduler_type == 'plateau':
                        # ReduceLROnPlateau needs validation loss
                        self.scheduler.step(val_loss)
                    elif self.scheduler_type in ['cosine', 'cosine_restarts', 'step']:
                        # These schedulers step automatically
                        self.scheduler.step()
                    
                    new_lr = self.optimizer.param_groups[0]['lr']
                    
                    # Log LR changes (for cosine, log every 10 epochs to avoid spam)
                    if self.scheduler_type == 'plateau' and old_lr != new_lr:
                        self.logger.info(f"🔻 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
                    elif epoch % 10 == 0:  # Log every 10 epochs for cosine
                        self.logger.info(f"📉 Learning rate: {new_lr:.6f}")
                
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
                
                # Determine save conditions
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                is_periodic_save = (epoch % self.save_frequency == 0)
                is_final_epoch = (epoch == self.num_epochs - 1)
                is_milestone = (epoch % 25 == 0)  # Extra saves at milestones
                
                # Save checkpoint with multiple conditions
                should_save = (
                    is_best or 
                    is_periodic_save or 
                    is_final_epoch or 
                    is_milestone or
                    (self.save_on_improvement and is_best)
                )
                
                if should_save:
                    save_reason = []
                    if is_best:
                        save_reason.append("best")
                    if is_periodic_save:
                        save_reason.append("periodic")
                    if is_final_epoch:
                        save_reason.append("final")
                    if is_milestone:
                        save_reason.append("milestone")
                    
                    self.logger.info(f"💾 Saving checkpoint - Reason: {', '.join(save_reason)}")
                    self.save_checkpoint(epoch, is_best=is_best, is_periodic=is_periodic_save)
                
                # Log save status every 10 epochs
                
                if epoch % 10 == 0:
                    self.logger.info(f"📊 Checkpoints saved: {len(self.saved_checkpoints)} periodic + best/latest")
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            # Save checkpoint on interruption
            self.logger.info("💾 Saving checkpoint due to interruption...")
            self.save_checkpoint(self.current_epoch, is_best=False, is_periodic=True)
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            # Save checkpoint on error for debugging
            self.logger.info("💾 Saving checkpoint due to error...")
            self.save_checkpoint(self.current_epoch, is_best=False, is_periodic=True)
            raise
        
        self.logger.info("Training completed")
        
        # Final checkpoint save
        self.save_checkpoint(self.current_epoch, is_best=False, is_periodic=True)
        
        # List all saved checkpoints
        self.list_saved_checkpoints()
        
        # Final health report
        final_train_kl = self.train_metrics.get('kl_loss', [0])[-1] if self.train_metrics.get('kl_loss') else 0
        final_val_kl = self.val_metrics.get('kl_loss', [0])[-1] if self.val_metrics.get('kl_loss') else 0
        
        self.logger.info(f"\n🎯 FINAL TRAINING REPORT:")
        self.logger.info(f"  Final KL losses - Train: {final_train_kl:.4f}, Val: {final_val_kl:.4f}")
        self.logger.info(f"  Target KL loss: > {self.target_kl_loss}")
        self.logger.info(f"  💾 Total checkpoints saved: {len(self.saved_checkpoints)} periodic + 2 special")
        
        if final_train_kl > self.target_kl_loss and final_val_kl > self.target_kl_loss:
            self.logger.info(f"  ✅ SUCCESS: Maintained healthy KL losses!")
            self.logger.info(f"  🎉 VAE should have meaningful latent space for HPRL")
        elif self.kl_collapse_detected:
            self.logger.error(f"  ❌ FAILURE: KL collapse detected during training")
            self.logger.error(f"  💡 Try: Lower beta, lower learning rate, or different architecture")
        else:
            self.logger.warning(f"  ⚠️ PARTIAL: KL losses below target but no collapse detected")
            self.logger.warning(f"  💡 Consider: Adjust beta coefficient for next training")
    
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