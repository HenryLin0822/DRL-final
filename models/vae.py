"""
FIXED VAE Model for HPRL - Anti-Collapse Version

Key fixes:
1. KL loss monitoring and enforcement
2. Improved reparameterization with clamping
3. Better loss computation with minimum KL enforcement
4. Active dimension tracking
5. Collapse prevention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class NNBase(nn.Module):
    """Base class for recurrent neural networks."""
    
    def __init__(self, recurrent, input_size, hidden_size, dropout=0.0, rnn_type='GRU'):
        super(NNBase, self).__init__()
        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self.rnn_type = rnn_type

        if recurrent:
            if rnn_type == 'GRU':
                self.rnn = nn.GRU(input_size, hidden_size, dropout=dropout)
            elif rnn_type == 'LSTM':
                self.rnn = nn.LSTM(input_size, hidden_size, dropout=dropout)
            else:
                raise NotImplementedError(f"RNN type {rnn_type} not supported")

            # Initialize parameters
            for name, param in self.rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def hidden_size(self):
        return self._hidden_size

    def forward_rnn(self, x, hxs, masks=None):
        """Forward pass through RNN with proper masking."""
        if self.rnn_type == 'GRU':
            if masks is not None:
                hxs = hxs * masks
            x, hxs = self.rnn(x.unsqueeze(0), hxs.unsqueeze(0))
            return x.squeeze(0), hxs.squeeze(0)
        elif self.rnn_type == 'LSTM':
            if masks is not None:
                hxs = (hxs[0] * masks, hxs[1] * masks)
            x, hxs = self.rnn(x.unsqueeze(0), (hxs[0].unsqueeze(0), hxs[1].unsqueeze(0)))
            return x.squeeze(0), (hxs[0].squeeze(0), hxs[1].squeeze(0))


class ProgramEncoder(NNBase):
    """Encoder that converts programs to latent representations."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, dropout=0.0, rnn_type='GRU'):
        super(ProgramEncoder, self).__init__(True, embedding_dim, hidden_size, dropout, rnn_type)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, programs, program_lengths):
        """
        Args:
            programs: [batch_size, seq_len] - tokenized programs
            program_lengths: [batch_size] - lengths of programs
        Returns:
            final_hidden: [batch_size, hidden_size] - encoded representations
        """
        embeddings = self.embedding(programs)
        program_lengths = program_lengths.cpu()
        
        packed = pack_padded_sequence(embeddings, program_lengths, 
                                    batch_first=True, enforce_sorted=False)
        
        if self.is_recurrent:
            _, final_hidden = self.rnn(packed)
            
        # Extract final hidden state based on RNN type
        if self.rnn_type == 'LSTM':
            final_hidden = final_hidden[0]  # Use hidden state, ignore cell state
            
        return final_hidden.squeeze(0)


class ProgramDecoder(NNBase):
    """Decoder that generates programs from latent representations."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, latent_dim, 
                 max_length=50, dropout=0.0, rnn_type='GRU'):
        input_size = embedding_dim + latent_dim
        super(ProgramDecoder, self).__init__(True, input_size, hidden_size, dropout, rnn_type)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.max_length = max_length
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Project latent to hidden size for RNN initialization
        self.latent_to_hidden = nn.Linear(latent_dim, self.hidden_size)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size + embedding_dim + latent_dim, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, vocab_size)
        )
        
    def forward_step(self, current_token, latent, hidden_state, mask=None):
        """Single forward step of decoder."""
        token_embed = self.embedding(current_token)
        rnn_input = torch.cat([token_embed, latent], dim=-1)
        
        rnn_output, new_hidden = self.forward_rnn(rnn_input, hidden_state, mask)
        
        # Project to vocabulary
        projection_input = torch.cat([rnn_output, token_embed, latent], dim=-1)
        logits = self.output_projection(projection_input)
        
        return logits, new_hidden
    
    def forward(self, latent, target_programs=None, max_length=None, deterministic=True):
        """
        Args:
            latent: [batch_size, latent_dim] - latent representations
            target_programs: [batch_size, seq_len] - target programs for teacher forcing
            max_length: maximum generation length
            deterministic: whether to use greedy decoding
        """
        batch_size = latent.size(0)
        device = latent.device
        max_length = max_length or self.max_length
        
        # Initialize hidden state by projecting latent to hidden size
        initial_hidden = self.latent_to_hidden(latent)
        current_token = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        if self.rnn_type == 'GRU':
            hidden_state = initial_hidden
        else:  # LSTM
            hidden_state = (initial_hidden, initial_hidden)
            
        outputs = []
        log_probs = []
        
        for step in range(max_length):
            logits, hidden_state = self.forward_step(current_token, latent, hidden_state)
            outputs.append(logits)
            
            # Get next token
            if target_programs is not None and step < target_programs.size(1) - 1:
                # Teacher forcing
                current_token = target_programs[:, step + 1]
            else:
                # Generate
                if deterministic:
                    current_token = logits.argmax(dim=-1)
                else:
                    probs = F.softmax(logits, dim=-1)
                    current_token = torch.multinomial(probs, 1).squeeze(-1)
                    
            # Calculate log probabilities for training
            log_prob = F.log_softmax(logits, dim=-1)
            log_probs.append(log_prob.gather(1, current_token.unsqueeze(1)))
            
        output_logits = torch.stack(outputs, dim=1)  # [batch_size, seq_len, vocab_size]
        output_log_probs = torch.cat(log_probs, dim=1)  # [batch_size, seq_len]
        
        return output_logits, output_log_probs


class StateEncoder(nn.Module):
    """CNN encoder for environment states."""
    
    def __init__(self, input_channels, height, width, hidden_size):
        super(StateEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, hidden_size),
            nn.ReLU()
        )
        
    def forward(self, states):
        return self.conv_layers(states)


class ConditionPolicy(NNBase):
    """Policy network conditioned on states and program latents."""
    
    def __init__(self, state_shape, num_actions, latent_dim, hidden_size, 
                 max_demo_length=10, dropout=0.0, rnn_type='GRU'):
        # Input: state_embed + latent + action_embed
        input_size = hidden_size + latent_dim + num_actions
        super(ConditionPolicy, self).__init__(True, input_size, hidden_size, dropout, rnn_type)
        
        self.num_actions = num_actions
        self.max_demo_length = max_demo_length
        
        self.state_encoder = StateEncoder(*state_shape, hidden_size)
        self.action_embedding = nn.Embedding(num_actions, num_actions)
        
        # Project latent to hidden size for RNN initialization
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_size)
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions)
        )
        
    def forward(self, states, latents, target_actions=None, deterministic=True):
        """
        Args:
            states: [batch_size, num_demos, seq_len, C, H, W] - environment states
            latents: [batch_size, latent_dim] - program latents
            target_actions: [batch_size, num_demos, seq_len-1] - target actions for training
        """
        batch_size, num_demos, seq_len = states.shape[:3]
        device = states.device
        
        # Encode initial states
        initial_states = states[:, :, 0]  # [batch_size, num_demos, C, H, W]
        state_embeds = self.state_encoder(initial_states.reshape(-1, *states.shape[3:]))
        state_embeds = state_embeds.reshape(batch_size, num_demos, -1)
        
        # Expand latents for each demo and project to hidden size
        latent_embeds = latents.unsqueeze(1).expand(-1, num_demos, -1).contiguous()
        hidden_init = self.latent_to_hidden(latent_embeds.reshape(-1, latent_embeds.size(-1)))
        
        # Initialize hidden states and actions
        if self.rnn_type == 'GRU':
            hidden_states = hidden_init
        else:  # LSTM
            hidden_states = (hidden_init, hidden_init)
            
        current_actions = torch.full((batch_size * num_demos,), self.num_actions - 1, 
                                   dtype=torch.long, device=device)
        
        action_logits = []
        action_log_probs = []
        
        for step in range(self.max_demo_length - 1):
            # Prepare inputs
            action_embeds = self.action_embedding(current_actions)
            action_embeds = action_embeds.reshape(batch_size, num_demos, -1)
            
            rnn_input = torch.cat([latent_embeds, state_embeds, action_embeds], dim=-1)
            rnn_input = rnn_input.reshape(-1, rnn_input.size(-1))
            
            # Forward through RNN
            rnn_output, hidden_states = self.forward_rnn(rnn_input, hidden_states)
            
            # Get action logits
            logits = self.action_head(rnn_output)
            action_logits.append(logits.reshape(batch_size, num_demos, -1))
            
            # Sample or use target actions
            if target_actions is not None and step < target_actions.size(-1):
                current_actions = target_actions[:, :, step].reshape(-1)
            else:
                if deterministic:
                    current_actions = logits.argmax(dim=-1)
                else:
                    probs = F.softmax(logits, dim=-1)
                    current_actions = torch.multinomial(probs, 1).squeeze(-1)
                    
            # Calculate log probabilities
            log_prob = F.log_softmax(logits, dim=-1)
            action_log_probs.append(log_prob.gather(1, current_actions.unsqueeze(1)))
            
        action_logits = torch.stack(action_logits, dim=-1)  # [batch_size, num_demos, num_actions, seq_len-1]
        action_log_probs = torch.cat(action_log_probs, dim=1)  # [batch_size*num_demos, seq_len-1]
        action_log_probs = action_log_probs.reshape(batch_size, num_demos, -1)
        
        return action_logits, action_log_probs


class VAE(nn.Module):
    """FIXED Variational Autoencoder for programs with collapse prevention."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, latent_dim, 
                 max_program_length=50, dropout=0.0, rnn_type='GRU'):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = ProgramEncoder(vocab_size, embedding_dim, hidden_size, dropout, rnn_type)
        self.decoder = ProgramDecoder(vocab_size, embedding_dim, hidden_size, latent_dim, 
                                    max_program_length, dropout, rnn_type)
        
        # Latent space projections
        self.mu_projection = nn.Linear(hidden_size, latent_dim)
        self.logvar_projection = nn.Linear(hidden_size, latent_dim)
        
        # ADDED: Diagnostics for monitoring
        self.last_kl_per_dim = None
        self.last_active_dims = 0
        self.last_mu_stats = {}
        self.last_logvar_stats = {}
        
    def encode(self, programs, program_lengths):
        """Encode programs to latent parameters."""
        encoded = self.encoder(programs, program_lengths)
        mu = self.mu_projection(encoded)
        logvar = self.logvar_projection(encoded)
        
        # ADDED: Store statistics for monitoring
        with torch.no_grad():
            self.last_mu_stats = {
                'mean': float(mu.mean().item()),
                'std': float(mu.std().item()),
                'norm': float(torch.norm(mu, dim=1).mean().item())
            }
            self.last_logvar_stats = {
                'mean': float(logvar.mean().item()),
                'std': float(logvar.std().item())
            }
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """FIXED: Reparameterization trick with clamping to prevent collapse."""
        std = torch.exp(0.5 * logvar)
        
        # CRITICAL: Clamp std to prevent collapse
        std = torch.clamp(std, min=1e-3, max=10.0)
        
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # ADDED: Monitor latent statistics
        with torch.no_grad():
            z_norm = torch.norm(z, dim=1).mean().item()
            if hasattr(self, 'last_z_norm'):
                self.last_z_norm = z_norm
        
        return z
    
    def decode(self, latent, target_programs=None, deterministic=True):
        """Decode latent variables to programs."""
        return self.decoder(latent, target_programs, deterministic=deterministic)
    
    def forward(self, programs, program_lengths, target_programs=None, deterministic=True):
        """Full VAE forward pass."""
        # Encode
        mu, logvar = self.encode(programs, program_lengths)
        
        # Sample latent
        latent = self.reparameterize(mu, logvar)
        
        # Decode
        output_logits, output_log_probs = self.decode(latent, target_programs, deterministic)
        
        return output_logits, output_log_probs, latent, mu, logvar
    
    def kl_loss(self, mu, logvar):
        """FIXED: KL divergence loss with monitoring and collapse prevention."""
        # Standard KL loss
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = kl_per_sample.mean()
        
        # ADDED: Monitor individual KL terms for diagnostics
        with torch.no_grad():
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)
            active_dims = (kl_per_dim > 0.01).sum().item()
            
            # Store diagnostics
            self.last_kl_per_dim = kl_per_dim
            self.last_active_dims = active_dims
            
            # Monitor mu and logvar variance across dimensions
            mu_var_across_dims = torch.var(mu, dim=0).mean().item()
            logvar_var_across_dims = torch.var(logvar, dim=0).mean().item()
            
            # Store additional diagnostics
            if not hasattr(self, 'kl_diagnostics'):
                self.kl_diagnostics = {}
            
            self.kl_diagnostics.update({
                'kl_per_sample_mean': float(kl_per_sample.mean().item()),
                'kl_per_sample_std': float(kl_per_sample.std().item()),
                'active_dimensions': active_dims,
                'total_dimensions': mu.size(1),
                'mu_variance_across_dims': mu_var_across_dims,
                'logvar_variance_across_dims': logvar_var_across_dims,
                'dimension_usage_ratio': active_dims / mu.size(1)
            })
        
        return kl_loss


class ProgramVAE(nn.Module):
    """
    FIXED ProgramVAE with collapse prevention and monitoring
    
    This model consists of:
    1. A VAE that learns to encode/decode programs with collapse prevention
    2. A condition policy that generates actions conditioned on program latents and states
    """
    
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=256, latent_dim=64,
                 state_shape=(3, 8, 8), num_actions=6, max_program_length=50, 
                 max_demo_length=10, dropout=0.0, rnn_type='GRU'):
        super(ProgramVAE, self).__init__()
        
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.max_program_length = max_program_length
        self.max_demo_length = max_demo_length
        
        # VAE for program synthesis with collapse prevention
        self.vae = VAE(vocab_size, embedding_dim, hidden_size, latent_dim, 
                      max_program_length, dropout, rnn_type)
        
        # Condition policy for action generation
        self.condition_policy = ConditionPolicy(state_shape, num_actions, latent_dim, 
                                              hidden_size, max_demo_length, dropout, rnn_type)
        
        # ADDED: Minimum KL loss enforcement
        self.min_kl_loss = 0.01  # Minimum KL to maintain
        self.kl_regularization_weight = 1.0
        
    def forward(self, programs, program_lengths, states=None, target_actions=None, 
                target_programs=None, deterministic=True, compute_policy=True):
        """
        Forward pass through both VAE and condition policy.
        
        Args:
            programs: [batch_size, seq_len] - input programs
            program_lengths: [batch_size] - program lengths
            states: [batch_size, num_demos, seq_len, C, H, W] - environment states
            target_actions: [batch_size, num_demos, seq_len-1] - target actions
            target_programs: [batch_size, seq_len] - target programs for reconstruction
            deterministic: whether to use deterministic generation
            compute_policy: whether to compute condition policy outputs
        """
        # VAE forward pass
        program_logits, program_log_probs, latent, mu, logvar = self.vae(
            programs, program_lengths, target_programs, deterministic
        )
        
        results = {
            'program_logits': program_logits,
            'program_log_probs': program_log_probs,
            'latent': latent,
            'mu': mu,
            'logvar': logvar,
            'kl_loss': self.vae.kl_loss(mu, logvar)
        }
        
        # ADDED: Store VAE diagnostics
        if hasattr(self.vae, 'kl_diagnostics'):
            results['kl_diagnostics'] = self.vae.kl_diagnostics
        
        # Condition policy forward pass
        #print("states:", states)
        #print("compute_policy:", compute_policy)
        if compute_policy and states is not None:
            action_logits, action_log_probs = self.condition_policy(
                states, latent, target_actions, deterministic
            )
            results.update({
                'action_logits': action_logits,
                'action_log_probs': action_log_probs
            })
        
        return results
    
    def generate_program(self, latent, max_length=None, deterministic=True):
        """Generate a program from latent representation."""
        max_length = max_length or self.max_program_length
        logits, log_probs = self.vae.decode(latent, deterministic=deterministic)
        if deterministic:
            return logits.argmax(dim=-1)
        else:
            return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
    
    def encode_program(self, programs, program_lengths):
        """Encode programs to latent space."""
        mu, logvar = self.vae.encode(programs, program_lengths)
        return self.vae.reparameterize(mu, logvar)
    
    def loss(self, programs, program_lengths, target_programs, states=None, 
             target_actions=None, beta=1.0, action_weight=1.0):
        """
        FIXED: Compute total loss with KL collapse prevention
        
        Args:
            beta: weight for KL divergence loss
            action_weight: weight for action prediction loss
        """
        results = self.forward(programs, program_lengths, states, target_actions, target_programs)
        
        # Reconstruction loss - handle variable length sequences
        batch_size, seq_len, vocab_size = results['program_logits'].shape
        target_seq_len = target_programs.size(1)
        
        # Truncate or pad to match sequence lengths
        min_len = min(seq_len, target_seq_len)
        program_logits_truncated = results['program_logits'][:, :min_len, :]
        target_programs_truncated = target_programs[:, :min_len]
        
        # Create mask for valid positions based on program lengths
        mask = torch.arange(min_len, device=program_lengths.device).unsqueeze(0) < program_lengths.unsqueeze(1)
        
        # Only compute loss on valid positions
        valid_positions = mask.flatten()
        if valid_positions.sum() > 0:
            valid_logits = program_logits_truncated.reshape(-1, vocab_size)[valid_positions]
            valid_targets = target_programs_truncated.flatten()[valid_positions]
            recon_loss = F.cross_entropy(valid_logits, valid_targets, reduction='mean')
        else:
            recon_loss = torch.tensor(0.0, device=programs.device)
        
        # KL divergence loss
        kl_loss = results['kl_loss']
        
        # CRITICAL: Enforce minimum KL loss to prevent collapse
        kl_regularization = torch.tensor(0.0, device=programs.device)
        if kl_loss < self.min_kl_loss and beta > 0.1:
            kl_regularization = self.kl_regularization_weight * (self.min_kl_loss - kl_loss)
            # Log this intervention
            if hasattr(self, '_log_kl_regularization'):
                self._log_kl_regularization = True
        
        total_loss = recon_loss + beta * kl_loss + kl_regularization
        
        # Action prediction loss (if applicable)
        action_loss = torch.tensor(0.0, device=programs.device)
        #print(results['action_logits'])
        #print("target_actions",target_actions)
        if 'action_logits' in results and target_actions is not None:
            # Debug shapes to understand the mismatch
            action_logits = results['action_logits']  # [batch_size, num_demos, num_actions, seq_len-1]
            
            # Transpose last two dimensions to get [batch_size, num_demos, seq_len-1, num_actions]
            action_logits_transposed = action_logits.transpose(-2, -1)
            
            # Get the actual sequence length from action logits
            actual_action_seq_len = action_logits_transposed.size(2)
            target_action_seq_len = target_actions.size(2)
            
            # Match sequence lengths
            min_action_len = min(actual_action_seq_len, target_action_seq_len)
            action_logits_matched = action_logits_transposed[:, :, :min_action_len, :]
            target_actions_matched = target_actions[:, :, :min_action_len]
            
            # Reshape for cross entropy: flatten batch and demo dimensions
            action_logits_flat = action_logits_matched.reshape(-1, action_logits_matched.size(-1))
            target_actions_flat = target_actions_matched.reshape(-1)
            
            action_loss = F.cross_entropy(
                action_logits_flat,
                target_actions_flat,
                reduction='mean'
            )
            #print("action weight:", action_weight)
            #print("action loss:", action_loss)
            total_loss += action_weight * action_loss
        
        loss_dict = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'action_loss': action_loss,
            'kl_regularization': kl_regularization
        }
        
        # ADDED: Include diagnostics in loss output
        if hasattr(self.vae, 'kl_diagnostics'):
            loss_dict['diagnostics'] = self.vae.kl_diagnostics.copy()
        
        return loss_dict
    
    def get_latent_space_health(self):
        """Get current health metrics of the latent space"""
        if not hasattr(self.vae, 'kl_diagnostics'):
            return {'error': 'No diagnostics available'}
        
        diagnostics = self.vae.kl_diagnostics
        
        health_status = 'unknown'
        if diagnostics.get('active_dimensions', 0) > self.latent_dim * 0.3:
            if diagnostics.get('dimension_usage_ratio', 0) > 0.5:
                health_status = 'healthy'
            else:
                health_status = 'moderate'
        else:
            health_status = 'poor'
        
        return {
            'health_status': health_status,
            'active_dimensions': diagnostics.get('active_dimensions', 0),
            'total_dimensions': self.latent_dim,
            'dimension_usage_ratio': diagnostics.get('dimension_usage_ratio', 0),
            'kl_per_sample_mean': diagnostics.get('kl_per_sample_mean', 0),
            'mu_variance': diagnostics.get('mu_variance_across_dims', 0),
            'logvar_variance': diagnostics.get('logvar_variance_across_dims', 0)
        }


# Example usage
if __name__ == "__main__":
    # Handle NumPy compatibility issue
    import warnings
    warnings.filterwarnings("ignore", message=".*NumPy.*")
    
    # Model parameters
    vocab_size = 100
    batch_size = 4
    seq_len = 20
    num_demos = 2
    demo_len = 8
    
    # Create model with anti-collapse features
    model = ProgramVAE(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_size=256,
        latent_dim=64,
        state_shape=(3, 8, 8),
        num_actions=6,
        max_program_length=50,
        max_demo_length=10
    )
    
    # Sample data
    programs = torch.randint(0, vocab_size, (batch_size, seq_len))
    program_lengths = torch.randint(5, seq_len, (batch_size,))
    states = torch.randn(batch_size, num_demos, demo_len, 3, 8, 8)
    target_actions = torch.randint(0, 6, (batch_size, num_demos, demo_len-1))
    
    print(f"Input shapes:")
    print(f"Programs: {programs.shape}")
    print(f"Program lengths: {program_lengths.shape}")
    print(f"States: {states.shape}")
    print(f"Target actions: {target_actions.shape}")
    
    # Forward pass
    results = model(programs, program_lengths, states, target_actions, programs)
    
    print(f"\nOutput shapes:")
    print(f"Program logits: {results['program_logits'].shape}")
    if 'action_logits' in results:
        print(f"Action logits: {results['action_logits'].shape}")
    
    # Compute loss with high beta to prevent collapse
    losses = model.loss(programs, program_lengths, programs, states, target_actions, 
                       beta=0.5, action_weight=1.0)  # High beta!
    
    print(f"\nFixed Model created successfully!")
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Reconstruction loss: {losses['recon_loss'].item()}")