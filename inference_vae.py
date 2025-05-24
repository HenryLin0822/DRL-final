#!/usr/bin/env python3
"""
HPRL VAE Inference Script

This script loads a trained ProgramVAE model and performs inference to evaluate
program reconstruction quality, latent space properties, and behavioral consistency.

Usage:
    python inference_vae.py --checkpoint path/to/model.pt [--input-programs path/to/programs.txt] [--num-samples 10]
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import seaborn as sns

# Add project modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vae import ProgramVAE
from dsl.karel_dsl import get_DSL_option_v2
from dsl.tokens import get_vocab_size, get_padding_index
from environments.karel_env import KarelEnvironment
from utils.config import config


class VAEInference:
    """
    Inference engine for trained ProgramVAE models
    
    Provides capabilities for:
    1. Program reconstruction evaluation
    2. Latent space analysis
    3. Program generation from latents
    4. Behavioral consistency testing
    5. Interpolation in latent space
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'auto',
        config_override: Optional[Dict] = None
    ):
        """
        Initialize VAE inference engine
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on ('auto', 'cuda', 'cpu')
            config_override: Optional config overrides
        """
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        
        # Load checkpoint and extract config
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = self.checkpoint.get('config', config)
        
        # Apply config overrides
        if config_override:
            self.config.update(config_override)
        
        # Initialize DSL and vocabulary
        self.dsl = get_DSL_option_v2(seed=self.config.get('seed', 42))
        self.vocab_size = get_vocab_size()
        self.padding_idx = get_padding_index()
        
        # Extract model parameters from config
        self.embedding_dim = 64
        self.hidden_size = self.config.get('net', {}).get('num_rnn_encoder_units', 256)
        self.latent_dim = 64
        self.max_program_length = self.config.get('dsl', {}).get('max_program_len', 12) + 1
        self.max_demo_length = self.config.get('rl', {}).get('envs', {}).get('executable', {}).get('max_demo_length', 100)
        
        # Initialize and load model
        self._load_model()
        
        # Initialize Karel environment for execution testing
        self.karel_env = KarelEnvironment()
        
        print(f"VAE Inference initialized successfully")
        print(f"Model trained for {self.checkpoint['epoch']} epochs")
        print(f"Best validation loss: {self.checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    def _setup_device(self, device_arg: str) -> str:
        """Setup inference device"""
        if device_arg == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device_arg
        
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        
        return device
    
    def _load_model(self):
        """Load the trained ProgramVAE model"""
        # Get number of actions from config
        num_actions = self.config.get('dsl', {}).get('num_agent_actions', 6)
        
        # Initialize ProgramVAE with same architecture as training
        self.model = ProgramVAE(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            latent_dim=self.latent_dim,
            state_shape=(8, 8, 8),  # Karel environment shape
            num_actions=num_actions,
            max_program_length=self.max_program_length,
            max_demo_length=self.max_demo_length,
            dropout=self.config['net']['dropout'],
            rnn_type=self.config['net']['rnn_type']
        ).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(self.checkpoint['program_vae_state_dict'])
        self.model.eval()
        
        print(f"Loaded ProgramVAE with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def tokenize_program(self, program_str: str) -> Tuple[torch.Tensor, int]:
        """
        Tokenize a program string into tensor format
        
        Args:
            program_str: Program as string
            
        Returns:
            tokens: [seq_len] tensor of token IDs
            length: Actual program length
        """
        tokens = self.dsl.str2intseq(program_str)
        
        # Pad to max length
        if len(tokens) < self.max_program_length:
            tokens = tokens + [self.padding_idx] * (self.max_program_length - len(tokens))
        else:
            tokens = tokens[:self.max_program_length]
        
        length = min(len(self.dsl.str2intseq(program_str)), self.max_program_length)
        
        return torch.tensor(tokens, dtype=torch.long), length
    
    def detokenize_program(self, tokens: torch.Tensor) -> str:
        """
        Convert token tensor back to program string
        
        Args:
            tokens: [seq_len] tensor of token IDs
            
        Returns:
            Program string
        """
        # Remove padding tokens
        tokens_list = tokens.cpu().numpy().tolist()
        tokens_list = [t for t in tokens_list if t != self.padding_idx]
        
        # Convert to string
        return self.dsl.intseq2str(tokens_list)
    
    def reconstruct_program(
        self, 
        program_str: str, 
        num_samples: int = 1,
        return_latent: bool = False
    ) -> Dict[str, Any]:
        """
        Reconstruct a program through the VAE
        
        Args:
            program_str: Input program as string
            num_samples: Number of reconstruction samples
            return_latent: Whether to return latent representation
            
        Returns:
            Dictionary containing reconstruction results
        """
        # Tokenize input program
        tokens, length = self.tokenize_program(program_str)
        tokens = tokens.unsqueeze(0).to(self.device)  # [1, seq_len]
        lengths = torch.tensor([length], device=self.device)
        
        results = []
        latents = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Use VAE encode method directly
                mu, logvar = self.model.vae.encode(tokens, lengths)
                latent = self.model.vae.reparameterize(mu, logvar)
                latents.append(latent)
                
                # Decode from latent - get output logits and log_probs
                output_logits, output_log_probs = self.model.vae.decode(latent, target_programs=tokens, deterministic=False)
                
                # Sample tokens from output logits
                reconstructed_tokens = []
                for step in range(min(output_logits.size(1), self.max_program_length)):
                    logits = output_logits[0, step, :]  # [vocab_size]
                    temperature = 0.7  # Lower = more conservative, higher = more diverse
                    probs = F.softmax(logits / temperature, dim=0)
                    if probs.sum() > 0:  # Ensure valid probability distribution
                        token = torch.multinomial(probs, 1).item()
                    else:
                        token = torch.argmax(logits).item()
                    reconstructed_tokens.append(token)
                    
                    # Stop at padding token
                    if token == self.padding_idx:
                        break
                
                # Convert back to string
                reconstructed_str = self.detokenize_program(torch.tensor(reconstructed_tokens))
                results.append(reconstructed_str)
        
        output = {
            'input_program': program_str,
            'reconstructed_programs': results,
            'reconstruction_diversity': len(set(results)),
            'exact_match_rate': sum(1 for r in results if r == program_str) / len(results)
        }
        
        if return_latent:
            output['latents'] = torch.cat(latents, dim=0)
        
        return output
    
    def evaluate_reconstruction_metrics(
        self, 
        programs: List[str], 
        num_samples_per_program: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate reconstruction quality on a list of programs
        
        Args:
            programs: List of program strings to evaluate
            num_samples_per_program: Number of reconstructions per program
            
        Returns:
            Dictionary of evaluation metrics
        """
        total_exact_matches = 0
        total_samples = 0
        total_diversity = 0
        syntax_error_count = 0
        
        print(f"Evaluating reconstruction on {len(programs)} programs...")
        
        for program in tqdm(programs):
            try:
                result = self.reconstruct_program(
                    program, 
                    num_samples=num_samples_per_program
                )
                
                total_exact_matches += result['exact_match_rate'] * num_samples_per_program
                total_samples += num_samples_per_program
                total_diversity += result['reconstruction_diversity']
                
                # Check for syntax errors in reconstructions
                for recon in result['reconstructed_programs']:
                    try:
                        # Try to parse the reconstructed program
                        self.dsl.str2intseq(recon)
                    except:
                        syntax_error_count += 1
                        
            except Exception as e:
                print(f"Error processing program '{program}': {e}")
                continue
        
        metrics = {
            'exact_match_rate': total_exact_matches / total_samples if total_samples > 0 else 0,
            'average_diversity': total_diversity / len(programs) if programs else 0,
            'syntax_error_rate': syntax_error_count / total_samples if total_samples > 0 else 0,
            'total_programs_evaluated': len(programs),
            'total_reconstructions': total_samples
        }
        
        return metrics
    
    def generate_from_prior(
        self, 
        num_samples: int = 10,
        temperature: float = 1.0
    ) -> List[str]:
        """
        Generate programs by sampling from the latent prior
        
        Args:
            num_samples: Number of programs to generate
            temperature: Sampling temperature (higher = more diverse)
            
        Returns:
            List of generated program strings
        """
        generated_programs = []
        
        with torch.no_grad():
            # Sample from standard normal prior
            latents = torch.randn(num_samples, self.latent_dim, device=self.device) * temperature
            
            # Decode from latents using VAE decoder
            output_logits, _ = self.model.vae.decode(latents, target_programs=None, deterministic=False)
            
            # Sample tokens from logits
            for i in range(num_samples):
                tokens = []
                for step in range(min(output_logits.size(1), self.max_program_length)):
                    logits = output_logits[i, step, :]  # [vocab_size]
                    probs = F.softmax(logits / temperature, dim=0)
                    if probs.sum() > 0:
                        token = torch.multinomial(probs, 1).item()
                    else:
                        token = torch.argmax(logits).item()
                    tokens.append(token)
                    
                    # Stop at padding token or end of sequence
                    if token == self.padding_idx:
                        break
                
                program_str = self.detokenize_program(torch.tensor(tokens))
                generated_programs.append(program_str)
        
        return generated_programs
    
    def interpolate_between_programs(
        self, 
        program1: str, 
        program2: str, 
        num_steps: int = 10
    ) -> List[str]:
        """
        Interpolate between two programs in latent space
        
        Args:
            program1: First program string
            program2: Second program string  
            num_steps: Number of interpolation steps
            
        Returns:
            List of interpolated program strings
        """
        # Encode both programs
        tokens1, length1 = self.tokenize_program(program1)
        tokens2, length2 = self.tokenize_program(program2)
        
        tokens1 = tokens1.unsqueeze(0).to(self.device)
        tokens2 = tokens2.unsqueeze(0).to(self.device)
        lengths1 = torch.tensor([length1], device=self.device)
        lengths2 = torch.tensor([length2], device=self.device)
        
        with torch.no_grad():
            # Encode both programs to get latents
            mu1, logvar1 = self.model.vae.encode(tokens1, lengths1)
            mu2, logvar2 = self.model.vae.encode(tokens2, lengths2)
            
            latent1 = self.model.vae.reparameterize(mu1, logvar1)
            latent2 = self.model.vae.reparameterize(mu2, logvar2)
            
            # Linear interpolation in latent space
            interpolated_programs = []
            
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                interpolated_latent = (1 - alpha) * latent1 + alpha * latent2
                
                # Generate program from interpolated latent
                output_logits, _ = self.model.vae.decode(interpolated_latent, target_programs=None, deterministic=True)
                
                # Get most likely tokens (deterministic)
                generated_tokens = torch.argmax(output_logits[0], dim=-1)
                program_str = self.detokenize_program(generated_tokens)
                interpolated_programs.append(program_str)
        
        return interpolated_programs
    
    def analyze_latent_space(
        self, 
        programs: List[str],
        save_plot: bool = True,
        plot_path: str = 'latent_space_analysis.png'
    ) -> Dict[str, Any]:
        """
        Analyze the structure of the learned latent space
        
        Args:
            programs: List of programs to analyze
            save_plot: Whether to save visualization plots
            plot_path: Path to save plots
            
        Returns:
            Analysis results
        """
        print(f"Analyzing latent space for {len(programs)} programs...")
        
        latents = []
        valid_programs = []
        
        # Encode all programs
        for program in tqdm(programs):
            try:
                tokens, length = self.tokenize_program(program)
                tokens = tokens.unsqueeze(0).to(self.device)
                lengths = torch.tensor([length], device=self.device)
                
                with torch.no_grad():
                    # Use VAE encode method directly
                    mu, logvar = self.model.vae.encode(tokens, lengths)
                    latent = self.model.vae.reparameterize(mu, logvar)
                    latents.append(latent.cpu().numpy())
                    valid_programs.append(program)
                    
            except Exception as e:
                print(f"Error encoding program '{program}': {e}")
                continue
        
        if not latents:
            return {'error': 'No valid programs to analyze'}
        
        latents = np.vstack(latents)  # [num_programs, latent_dim]
        
        # Compute statistics
        latent_mean = np.mean(latents, axis=0)
        latent_std = np.std(latents, axis=0)
        latent_norm = np.linalg.norm(latents, axis=1)
        
        # Compute pairwise distances
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(latents)
        
        analysis = {
            'num_programs': len(valid_programs),
            'latent_dim': latents.shape[1],
            'mean_latent_norm': float(np.mean(latent_norm)),
            'std_latent_norm': float(np.std(latent_norm)),
            'mean_pairwise_distance': float(np.mean(distances[np.triu_indices_from(distances, k=1)])),
            'latent_dimension_usage': {
                'mean_activation': latent_mean.tolist(),
                'std_activation': latent_std.tolist(),
                'effective_dimensions': int(np.sum(latent_std > 0.1))  # Dimensions with significant variance
            }
        }
        
        # Create visualizations if requested
        if save_plot and len(valid_programs) > 1:
            self._create_latent_visualizations(latents, valid_programs, plot_path)
        
        return analysis
    
    def _create_latent_visualizations(
        self, 
        latents: np.ndarray, 
        programs: List[str], 
        plot_path: str
    ):
        """Create visualizations of the latent space"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Latent dimension activations
        axes[0, 0].boxplot([latents[:, i] for i in range(min(20, latents.shape[1]))])
        axes[0, 0].set_title('Latent Dimension Activations (First 20 dims)')
        axes[0, 0].set_xlabel('Latent Dimension')
        axes[0, 0].set_ylabel('Activation Value')
        
        # 2. Latent norm distribution
        axes[0, 1].hist(np.linalg.norm(latents, axis=1), bins=30, alpha=0.7)
        axes[0, 1].set_title('Distribution of Latent Vector Norms')
        axes[0, 1].set_xlabel('L2 Norm')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. PCA visualization (if enough samples)
        if latents.shape[0] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            latents_2d = pca.fit_transform(latents)
            
            axes[1, 0].scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.6)
            axes[1, 0].set_title(f'PCA Visualization (Explained Variance: {pca.explained_variance_ratio_.sum():.2f})')
            axes[1, 0].set_xlabel('PC1')
            axes[1, 0].set_ylabel('PC2')
        
        # 4. Pairwise distance heatmap (sample if too many programs)
        if latents.shape[0] > 50:
            # Sample 50 programs for visualization
            indices = np.random.choice(latents.shape[0], 50, replace=False)
            sample_latents = latents[indices]
        else:
            sample_latents = latents
        
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(sample_latents)
        
        im = axes[1, 1].imshow(distances, cmap='viridis')
        axes[1, 1].set_title('Pairwise Latent Distances')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Latent space visualizations saved to {plot_path}")
    
    def test_behavioral_consistency(
        self, 
        programs: List[str],
        num_test_states: int = 5
    ) -> Dict[str, Any]:
        """
        Test if similar programs have similar latent representations
        
        Args:
            programs: List of programs to test
            num_test_states: Number of random states to test execution on
            
        Returns:
            Behavioral consistency analysis
        """
        print(f"Testing behavioral consistency for {len(programs)} programs...")
        
        # This would require implementing program execution
        # For now, return a placeholder structure
        
        return {
            'message': 'Behavioral consistency testing requires program execution implementation',
            'num_programs_tested': len(programs),
            'note': 'This would compare execution traces and latent similarities'
        }
    
    def run_comprehensive_evaluation(
        self, 
        test_programs: List[str],
        output_dir: str = './inference_results'
    ) -> Dict[str, Any]:
        """
        Run a comprehensive evaluation of the VAE
        
        Args:
            test_programs: List of programs to evaluate on
            output_dir: Directory to save results
            
        Returns:
            Complete evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("Running comprehensive VAE evaluation...")
        
        results = {}
        
        # 1. Reconstruction quality
        print("\n1. Evaluating reconstruction quality...")
        recon_metrics = self.evaluate_reconstruction_metrics(test_programs)
        results['reconstruction_metrics'] = recon_metrics
        
        # 2. Prior sampling quality
        print("\n2. Evaluating prior sampling...")
        generated_programs = self.generate_from_prior(num_samples=20)
        results['generated_programs'] = generated_programs
        
        # 3. Latent space analysis
        print("\n3. Analyzing latent space...")
        latent_analysis = self.analyze_latent_space(
            test_programs, 
            save_plot=True,
            plot_path=os.path.join(output_dir, 'latent_analysis.png')
        )
        results['latent_analysis'] = latent_analysis
        
        # 4. Interpolation examples
        print("\n4. Testing interpolation...")
        if len(test_programs) >= 2:
            interpolation = self.interpolate_between_programs(
                test_programs[0], 
                test_programs[1], 
                num_steps=5
            )
            results['interpolation_example'] = {
                'program1': test_programs[0],
                'program2': test_programs[1],
                'interpolated_programs': interpolation
            }
        
        # 5. Individual reconstruction examples
        print("\n5. Getting reconstruction examples...")
        reconstruction_examples = []
        for i, program in enumerate(test_programs[:5]):  # First 5 programs
            recon_result = self.reconstruct_program(program, num_samples=3)
            reconstruction_examples.append(recon_result)
        results['reconstruction_examples'] = reconstruction_examples
        
        # Save results
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nEvaluation completed! Results saved to {output_dir}")
        return results


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="VAE Inference and Evaluation")
    
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--input-programs', type=str, default=None,
                       help='Path to text file with input programs (one per line)')
    parser.add_argument('--programs', type=str, nargs='+', default=None,
                       help='Individual programs to test (space-separated)')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples for generation/reconstruction')
    parser.add_argument('--output-dir', type=str, default='./inference_results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--mode', type=str, default='comprehensive',
                       choices=['comprehensive', 'reconstruct', 'generate', 'interpolate'],
                       help='Inference mode')
    
    return parser.parse_args()


def load_programs_from_file(file_path: str) -> List[str]:
    """Load programs from a text file"""
    with open(file_path, 'r') as f:
        programs = [line.strip() for line in f if line.strip()]
    return programs


def main():
    """Main inference function"""
    args = parse_arguments()
    
    # Initialize inference engine
    print(f"Loading model from {args.checkpoint}...")
    inference = VAEInference(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Get test programs
    test_programs = []
    
    if args.input_programs:
        test_programs = load_programs_from_file(args.input_programs)
        print(f"Loaded {len(test_programs)} programs from {args.input_programs}")
    
    if args.programs:
        test_programs.extend(args.programs)
        print(f"Added {len(args.programs)} programs from command line")
    
    # Use default programs if none provided
    if not test_programs:
        test_programs = [
            "WHILE frontIsClear DO move END",
            "REPEAT 3 TIMES move END",
            "IF frontIsClear THEN move ELSE turnLeft END",
            "WHILE notFacingNorth DO turnLeft END"
        ]
        print(f"Using {len(test_programs)} default test programs")
    
    # Run inference based on mode
    if args.mode == 'comprehensive':
        results = inference.run_comprehensive_evaluation(test_programs, args.output_dir)
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        recon_metrics = results['reconstruction_metrics']
        print(f"Exact Match Rate: {recon_metrics['exact_match_rate']:.3f}")
        print(f"Syntax Error Rate: {recon_metrics['syntax_error_rate']:.3f}")
        print(f"Average Diversity: {recon_metrics['average_diversity']:.3f}")
        
        latent_analysis = results['latent_analysis']
        print(f"Mean Latent Norm: {latent_analysis['mean_latent_norm']:.3f}")
        print(f"Effective Dimensions: {latent_analysis['latent_dimension_usage']['effective_dimensions']}")
        
        print(f"\nGenerated Programs:")
        for i, prog in enumerate(results['generated_programs'][:5]):
            print(f"  {i+1}: {prog}")
    
    elif args.mode == 'reconstruct':
        print(f"\nReconstructing {len(test_programs)} programs...")
        for program in test_programs:
            result = inference.reconstruct_program(program, num_samples=args.num_samples)
            print(f"\nInput: {result['input_program']}")
            print(f"Exact Match Rate: {result['exact_match_rate']:.3f}")
            print("Reconstructions:")
            for i, recon in enumerate(result['reconstructed_programs'][:5]):
                print(f"  {i+1}: {recon}")
    
    elif args.mode == 'generate':
        print(f"\nGenerating {args.num_samples} programs from prior...")
        generated = inference.generate_from_prior(num_samples=args.num_samples)
        for i, program in enumerate(generated):
            print(f"  {i+1}: {program}")
    
    elif args.mode == 'interpolate':
        if len(test_programs) >= 2:
            print(f"\nInterpolating between programs...")
            interpolated = inference.interpolate_between_programs(
                test_programs[0], test_programs[1], num_steps=args.num_samples
            )
            print(f"From: {test_programs[0]}")
            print(f"To: {test_programs[1]}")
            print("Interpolation:")
            for i, program in enumerate(interpolated):
                print(f"  {i+1}: {program}")
        else:
            print("Need at least 2 programs for interpolation")


if __name__ == "__main__":
    main()