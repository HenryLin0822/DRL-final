#!/usr/bin/env python3
"""
IMPROVED VAE Inference Script - True Generation Capability

This script provides inference methods for VAE models trained WITHOUT teacher forcing.
Key features:
1. Pure generation from random latent vectors
2. True SLERP interpolation in latent space
3. Compatibility with both improved and legacy models
4. No teacher forcing during inference
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import math

# Add project modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vae import ProgramVAE
from dsl.karel_dsl import get_DSL_option_v2
from dsl.tokens import get_vocab_size, get_padding_index
from utils.config import config


def slerp(z1: torch.Tensor, z2: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical Linear Interpolation (SLERP) between two vectors.
    
    Args:
        z1: First vector [batch_size, latent_dim]
        z2: Second vector [batch_size, latent_dim] 
        t: Interpolation parameter [0, 1]
    
    Returns:
        Interpolated vector
    """
    # Normalize vectors
    z1_norm = F.normalize(z1, p=2, dim=-1)
    z2_norm = F.normalize(z2, p=2, dim=-1)
    
    # Compute angle between vectors
    dot_product = torch.sum(z1_norm * z2_norm, dim=-1, keepdim=True)
    # Clamp to avoid numerical issues
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Compute angle
    omega = torch.acos(dot_product)
    
    # Handle near-parallel vectors (fall back to linear interpolation)
    sin_omega = torch.sin(omega)
    parallel_mask = (sin_omega.abs() < 1e-6).squeeze(-1)
    
    if parallel_mask.any():
        # Linear interpolation for parallel vectors
        result = (1 - t) * z1 + t * z2
        return result
    
    # SLERP formula
    coeff1 = torch.sin((1 - t) * omega) / sin_omega
    coeff2 = torch.sin(t * omega) / sin_omega
    
    # Interpolate with original vector magnitudes
    z1_mag = torch.norm(z1, p=2, dim=-1, keepdim=True)
    z2_mag = torch.norm(z2, p=2, dim=-1, keepdim=True)
    interpolated_mag = (1 - t) * z1_mag + t * z2_mag
    
    interpolated_norm = coeff1 * z1_norm + coeff2 * z2_norm
    interpolated = interpolated_norm * interpolated_mag
    
    return interpolated


class ImprovedVAEInference:
    """
    Improved VAE Inference - True generation without teacher forcing
    
    Compatible with both improved models (trained without teacher forcing)
    and legacy models (with fallback methods)
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'auto',
        temperature: float = 1.0
    ):
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.temperature = temperature
        
        # Load checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = self.checkpoint.get('config', config)
        
        # Initialize DSL and vocabulary
        self.dsl = get_DSL_option_v2(seed=self.config.get('seed', 42))
        self.vocab_size = get_vocab_size()
        self.padding_idx = get_padding_index()
        
        # Model parameters
        self.embedding_dim = 64
        self.hidden_size = 256
        self.latent_dim = 64
        self.max_program_length = 13
        
        # Load model
        self._load_model()
        
        # Check model capabilities
        self.capabilities = self._check_model_capabilities()
        
        print(f"🚀 Improved VAE Inference initialized")
        print(f"🎯 Pure generation mode (no teacher forcing)")
        print(f"🌀 SLERP interpolation enabled")
        print(f"🔥 Temperature: {self.temperature}")
    
    def _setup_device(self, device_arg: str) -> str:
        if device_arg == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device_arg
        return device
    
    def _load_model(self):
        """Load the trained ProgramVAE model"""
        num_actions = self.config.get('dsl', {}).get('num_agent_actions', 6)
        
        self.model = ProgramVAE(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            latent_dim=self.latent_dim,
            state_shape=(8, 8, 8),
            num_actions=num_actions,
            max_program_length=self.max_program_length,
            max_demo_length=100,
            dropout=self.config['net']['dropout'],
            rnn_type=self.config['net']['rnn_type']
        ).to(self.device)
        
        self.model.load_state_dict(self.checkpoint['program_vae_state_dict'])
        self.model.eval()
        
        print(f"Model loaded: {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def _check_model_capabilities(self) -> Dict[str, Any]:
        """Check what generation capabilities the model has"""
        capabilities = {
            'has_pure_generation': hasattr(self.model, 'generate_program_pure'),
            'has_pure_loss': hasattr(self.model, 'pure_generation_loss'),
            'generation_success_rate': self.checkpoint.get('generation_success_rate', None),
            'training_method': 'improved' if self.checkpoint.get('generation_success_rate') is not None else 'legacy'
        }
        
        print(f"🔍 Model capabilities:")
        
        if capabilities['has_pure_generation']:
            print(f"   ✅ Pure generation method available")
        else:
            print(f"   ⚠️ Using fallback generation method")
            
        if capabilities['has_pure_loss']:
            print(f"   ✅ Pure generation loss method available")
        else:
            print(f"   ⚠️ No pure generation loss method")
            
        if capabilities['generation_success_rate'] is not None:
            print(f"   ✅ Generation success rate: {capabilities['generation_success_rate']:.2%}")
        else:
            print(f"   ⚠️ No generation training statistics available")
            
        print(f"   🏷️ Training method: {capabilities['training_method']}")
        
        return capabilities
    
    def tokenize_program(self, program_str: str) -> Tuple[torch.Tensor, int]:
        """Tokenize program exactly like training"""
        try:
            tokens = self.dsl.str2intseq(program_str)
        except Exception as e:
            print(f"Warning: Failed to tokenize '{program_str}': {e}")
            tokens = []
        
        original_length = len(tokens)
        
        # Pad exactly like training
        padded_tokens = tokens + [self.padding_idx] * (self.max_program_length - len(tokens))
        padded_tokens = padded_tokens[:self.max_program_length]
        
        length = min(original_length, self.max_program_length)
        length = max(1, length)
        
        return torch.tensor(padded_tokens, dtype=torch.long), length
    
    def detokenize_program(self, tokens: torch.Tensor) -> str:
        """Convert tokens back to program string"""
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
    
    def encode_program(self, program_str: str) -> torch.Tensor:
        """Encode a program to latent space"""
        tokens, length = self.tokenize_program(program_str)
        tokens_batch = tokens.unsqueeze(0).to(self.device)
        lengths_batch = torch.tensor([length], device=self.device)
        
        with torch.no_grad():
            mu, logvar = self.model.vae.encode(tokens_batch, lengths_batch)
            return mu
    
    def decode_latent(self, latent: torch.Tensor, deterministic: bool = True) -> str:
        """
        Decode latent vector to program WITHOUT teacher forcing
        
        Uses improved generation methods if available, with fallback
        """
        with torch.no_grad():
            try:
                # Method 1: Use improved pure generation if available
                if self.capabilities['has_pure_generation']:
                    generated_tokens = self.model.generate_program_pure(
                        latent, 
                        deterministic=deterministic,
                        temperature=self.temperature if not deterministic else 1.0
                    )
                    # Extract first sequence and convert to string
                    token_sequence = generated_tokens[0]
                    program_str = self.detokenize_program(token_sequence)
                    return program_str
                
                # Method 2: Fallback to standard decode WITHOUT teacher forcing
                else:
                    output_logits, _ = self.model.vae.decode(
                        latent, 
                        target_programs=None,  # KEY: No teacher forcing!
                        deterministic=deterministic
                    )
                    
                    if not deterministic:
                        # Apply temperature sampling
                        output_logits = output_logits / self.temperature
                        probs = F.softmax(output_logits, dim=-1)
                        generated_tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(output_logits.shape[:2])
                    else:
                        # Greedy decoding
                        generated_tokens = torch.argmax(output_logits, dim=-1)
                    
                    # Extract first sequence
                    token_sequence = generated_tokens[0]
                    program_str = self.detokenize_program(token_sequence)
                    return program_str
                    
            except Exception as e:
                return f"<DECODE_ERROR: {str(e)}>"
    
    def generate_programs(
        self, 
        num_samples: int = 10,
        deterministic: bool = True,
        latent_std: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Generate programs from random latent vectors (pure generation)
        
        Updated to work with improved VAE model without teacher forcing
        """
        print(f"🎲 Generating {num_samples} programs from random latent vectors...")
        print(f"   Deterministic: {deterministic}, Latent std: {latent_std}")
        print(f"   Method: {self.capabilities['training_method']}")
        
        generated_programs = []
        
        with torch.no_grad():
            # Sample random latent vectors
            latent_vectors = torch.randn(num_samples, self.latent_dim, device=self.device) * latent_std
            
            # Method 1: Batch generation if available
            if self.capabilities['has_pure_generation']:
                print("   Using improved pure generation method")
                try:
                    # Use batch generation for efficiency
                    generated_tokens = self.model.generate_program_pure(
                        latent_vectors, 
                        deterministic=deterministic,
                        temperature=self.temperature if not deterministic else 1.0
                    )
                    
                    for i in range(num_samples):
                        tokens = generated_tokens[i]
                        program_str = self.detokenize_program(tokens)
                        
                        result = {
                            'sample_id': i + 1,
                            'program': program_str,
                            'latent_vector': latent_vectors[i].cpu().numpy().tolist(),
                            'latent_norm': torch.norm(latent_vectors[i]).item(),
                            'deterministic': deterministic,
                            'method': 'pure_generation_batch'
                        }
                        
                        generated_programs.append(result)
                        print(f"  {i+1}: {program_str}")
                        
                except Exception as e:
                    print(f"   Batch generation failed: {e}, falling back to individual generation")
                    # Fallback to individual generation
                    for i in range(num_samples):
                        latent = latent_vectors[i:i+1]  # Keep batch dimension
                        program_str = self.decode_latent(latent, deterministic=deterministic)
                        
                        result = {
                            'sample_id': i + 1,
                            'program': program_str,
                            'latent_vector': latent.cpu().numpy().tolist(),
                            'latent_norm': torch.norm(latent).item(),
                            'deterministic': deterministic,
                            'method': 'pure_generation_individual'
                        }
                        
                        generated_programs.append(result)
                        print(f"  {i+1}: {program_str}")
            
            # Method 2: Individual generation
            else:
                print("   Using standard decode method")
                for i in range(num_samples):
                    latent = latent_vectors[i:i+1]  # Keep batch dimension
                    program_str = self.decode_latent(latent, deterministic=deterministic)
                    
                    result = {
                        'sample_id': i + 1,
                        'program': program_str,
                        'latent_vector': latent.cpu().numpy().tolist(),
                        'latent_norm': torch.norm(latent).item(),
                        'deterministic': deterministic,
                        'method': 'standard_decode'
                    }
                    
                    generated_programs.append(result)
                    print(f"  {i+1}: {program_str}")
        
        return generated_programs
    
    def slerp_interpolate(
        self, 
        program1: str, 
        program2: str, 
        num_steps: int = 7,
        deterministic: bool = True
    ) -> List[Dict[str, Any]]:
        """
        SLERP interpolation between two programs in latent space
        
        True interpolation using improved generation capability
        """
        print(f"🌀 SLERP interpolation between programs ({num_steps} steps)...")
        print(f"   From: {program1}")
        print(f"   To:   {program2}")
        print(f"   Deterministic: {deterministic}")
        print(f"   Method: {self.capabilities['training_method']}")
        
        # Encode both programs to latent space
        latent1 = self.encode_program(program1)
        latent2 = self.encode_program(program2)
        
        interpolated_programs = []
        
        with torch.no_grad():
            for i in range(num_steps):
                # Compute interpolation parameter
                t = i / (num_steps - 1) if num_steps > 1 else 0.0
                
                # SLERP interpolation
                interpolated_latent = slerp(latent1, latent2, t)
                
                # Decode interpolated latent WITHOUT teacher forcing
                program_str = self.decode_latent(interpolated_latent, deterministic=deterministic)
                
                result = {
                    'step': i + 1,
                    'interpolation_t': t,
                    'program': program_str,
                    'latent_vector': interpolated_latent.cpu().numpy().tolist(),
                    'latent_norm': torch.norm(interpolated_latent).item(),
                    'deterministic': deterministic,
                    'method': self.capabilities['training_method']
                }
                
                interpolated_programs.append(result)
                print(f"  {i+1} (t={t:.2f}): {program_str}")
        
        return interpolated_programs
    
    def latent_space_walk(
        self,
        start_program: str,
        direction: torch.Tensor = None,
        num_steps: int = 5,
        step_size: float = 0.5,
        deterministic: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Walk in latent space from a starting program
        
        True latent space exploration using improved generation
        """
        print(f"🚶 Latent space walk ({num_steps} steps, step_size={step_size})...")
        print(f"   Starting from: {start_program}")
        print(f"   Method: {self.capabilities['training_method']}")
        
        # Encode starting program
        start_latent = self.encode_program(start_program)
        
        # Generate random direction if not provided
        if direction is None:
            direction = torch.randn_like(start_latent)
            direction = F.normalize(direction, p=2, dim=-1)
        
        walk_programs = []
        
        with torch.no_grad():
            for i in range(num_steps):
                # Move in direction
                current_latent = start_latent + i * step_size * direction
                
                # Decode current position WITHOUT teacher forcing
                program_str = self.decode_latent(current_latent, deterministic=deterministic)
                
                result = {
                    'step': i + 1,
                    'distance': i * step_size,
                    'program': program_str,
                    'latent_vector': current_latent.cpu().numpy().tolist(),
                    'latent_norm': torch.norm(current_latent).item(),
                    'deterministic': deterministic,
                    'method': self.capabilities['training_method']
                }
                
                walk_programs.append(result)
                print(f"  {i+1} (dist={i * step_size:.2f}): {program_str}")
        
        return walk_programs
    
    def reconstruct_program(
        self, 
        program_str: str, 
        return_latent: bool = False
    ) -> Dict[str, Any]:
        """
        Reconstruct a program using the VAE (encode then decode)
        
        This tests the round-trip capability of the VAE
        """
        print(f"🔄 Reconstructing program: {program_str}")
        
        # Tokenize input program
        tokens, length = self.tokenize_program(program_str)
        tokens_batch = tokens.unsqueeze(0).to(self.device)
        lengths_batch = torch.tensor([length], device=self.device)
        
        with torch.no_grad():
            # Encode to latent space
            mu, logvar = self.model.vae.encode(tokens_batch, lengths_batch)
            latent = self.model.vae.reparameterize(mu, logvar)
            
            # Decode WITHOUT teacher forcing
            reconstructed_str = self.decode_latent(latent, deterministic=True)
            
            # Check if reconstruction is reasonable
            original_tokens = tokens.cpu().numpy()[:length]
            
            result = {
                'input_program': program_str,
                'reconstructed_program': reconstructed_str,
                'input_tokens': original_tokens.tolist(),
                'method': self.capabilities['training_method'],
                'latent_norm': torch.norm(latent).item()
            }
            
            if return_latent:
                result['latent'] = latent.cpu().numpy().tolist()
            
            print(f"   Reconstructed: {reconstructed_str}")
            
            return result
    
    def run_generation_tests(self, num_samples: int = 10) -> Dict[str, Any]:
        """Run comprehensive generation tests"""
        print(f"🧪 Running generation tests ({num_samples} samples)...")
        
        # Test 1: Pure generation
        print(f"\n1. Pure Generation Test:")
        generation_results = self.generate_programs(num_samples, deterministic=True)
        
        # Analyze generation quality
        valid_programs = 0
        parse_errors = 0
        empty_programs = 0
        
        for result in generation_results:
            program = result['program']
            if '<PARSE_ERROR' in program:
                parse_errors += 1
            elif program == '<EMPTY>':
                empty_programs += 1
            elif program.startswith('DEF'):
                valid_programs += 1
        
        generation_quality = {
            'total_samples': num_samples,
            'valid_programs': valid_programs,
            'parse_errors': parse_errors,
            'empty_programs': empty_programs,
            'success_rate': valid_programs / num_samples,
            'results': generation_results
        }
        
        # Test 2: Reconstruction test
        print(f"\n2. Reconstruction Test:")
        test_programs = [
            "DEF run m( move m)",
            "DEF run m( turnLeft m)",
            "DEF run m( turnRight m)",
            "DEF run m( pickMarker m)",
            "DEF run m( putMarker m)"
        ]
        
        reconstruction_results = []
        for program in test_programs:
            result = self.reconstruct_program(program)
            reconstruction_results.append(result)
        
        # Test 3: Interpolation test
        print(f"\n3. Interpolation Test:")
        if len(test_programs) >= 2:
            interpolation_results = self.slerp_interpolate(
                test_programs[0], test_programs[1], num_steps=5
            )
        else:
            interpolation_results = []
        
        # Summary
        summary = {
            'model_capabilities': self.capabilities,
            'generation_quality': generation_quality,
            'reconstruction_results': reconstruction_results,
            'interpolation_results': interpolation_results,
            'timestamp': torch.utils.data.get_worker_info(),
        }
        
        print(f"\n📊 Test Summary:")
        print(f"  Generation success rate: {generation_quality['success_rate']:.2%}")
        print(f"  Valid programs: {valid_programs}/{num_samples}")
        print(f"  Model type: {self.capabilities['training_method']}")
        
        if self.capabilities['generation_success_rate'] is not None:
            print(f"  Training success rate: {self.capabilities['generation_success_rate']:.2%}")
        
        return summary


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Improved VAE Inference - True Generation")
    
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--mode', type=str, default='generate',
                       choices=['generate', 'interpolate', 'walk', 'reconstruct', 'test'],
                       help='Inference mode')
    
    # Generation parameters
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to generate')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic (greedy) decoding')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (if not deterministic)')
    parser.add_argument('--latent-std', type=float, default=1.0,
                       help='Standard deviation for random latent vectors')
    
    # Interpolation parameters
    parser.add_argument('--programs', type=str, nargs='+', default=None,
                       help='Programs for interpolation or reconstruction')
    parser.add_argument('--num-steps', type=int, default=7,
                       help='Number of interpolation steps')
    
    # Walk parameters
    parser.add_argument('--step-size', type=float, default=0.5,
                       help='Step size for latent walk')
    
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    parser.add_argument('--save-results', type=str, default=None,
                       help='Path to save results JSON')
    
    return parser.parse_args()


def main():
    """Main inference function"""
    args = parse_arguments()
    
    # Initialize improved inference engine
    print(f"Loading improved model from {args.checkpoint}...")
    inference = ImprovedVAEInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        temperature=args.temperature
    )
    
    results = {}
    
    if args.mode == 'generate':
        print(f"\n🎲 Pure Generation Mode")
        results = inference.generate_programs(
            num_samples=args.num_samples,
            deterministic=args.deterministic,
            latent_std=args.latent_std
        )
    
    elif args.mode == 'interpolate':
        if not args.programs or len(args.programs) < 2:
            print("❌ Need at least 2 programs for interpolation")
            print("Example: --programs 'DEF run m( move m)' 'DEF run m( turnLeft m)'")
            return
        
        print(f"\n🌀 SLERP Interpolation Mode")
        results = inference.slerp_interpolate(
            program1=args.programs[0],
            program2=args.programs[1],
            num_steps=args.num_steps,
            deterministic=args.deterministic
        )
    
    elif args.mode == 'walk':
        if not args.programs or len(args.programs) < 1:
            print("❌ Need at least 1 program for latent walk")
            return
        
        print(f"\n🚶 Latent Space Walk Mode")
        results = inference.latent_space_walk(
            start_program=args.programs[0],
            num_steps=args.num_steps,
            step_size=args.step_size,
            deterministic=args.deterministic
        )
    
    elif args.mode == 'reconstruct':
        if not args.programs or len(args.programs) < 1:
            print("❌ Need at least 1 program for reconstruction")
            return
        
        print(f"\n🔄 Reconstruction Mode")
        results = []
        for program in args.programs:
            result = inference.reconstruct_program(program, return_latent=True)
            results.append(result)
    
    elif args.mode == 'test':
        print(f"\n🧪 Comprehensive Test Mode")
        results = inference.run_generation_tests(num_samples=args.num_samples)
    
    # Save results if requested
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.save_results}")
    
    print(f"\n✅ Improved VAE inference completed!")
    print(f"🚀 True generation capability - no teacher forcing!")


if __name__ == "__main__":
    main()