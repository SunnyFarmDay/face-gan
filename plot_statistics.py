#!/usr/bin/env python3
"""
Script to generate training visualizations from statistics CSV file.

This script reads training_statistics.csv and generates comprehensive plots
showing the training progress of the Progressive GAN.
"""

import csv
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_statistics(csv_file):
    """Load statistics from CSV file."""
    iterations = []
    gen_losses = []
    disc_losses = []
    w_losses = []
    grad_penalties = []
    alphas = []
    steps = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            iterations.append(int(row['iteration']))
            gen_losses.append(float(row['generator_loss']))
            disc_losses.append(float(row['discriminator_loss']))
            w_losses.append(float(row['wasserstein_loss']))
            grad_penalties.append(float(row['gradient_penalty']))
            alphas.append(float(row['alpha']))
            steps.append(int(row['step']))
    
    return {
        'iterations': iterations,
        'gen_losses': gen_losses,
        'disc_losses': disc_losses,
        'w_losses': w_losses,
        'grad_penalties': grad_penalties,
        'alphas': alphas,
        'steps': steps
    }


def plot_comprehensive(data, output_file='training_analysis.png', title_prefix=''):
    """Generate comprehensive training plots."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    iterations = data['iterations']
    
    # 1. Generator vs Discriminator vs Wasserstein Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iterations, data['gen_losses'], label='Generator', color='blue', alpha=0.7, linewidth=1.5)
    ax1.plot(iterations, data['disc_losses'], label='Discriminator', color='red', alpha=0.7, linewidth=1.5)
    ax1.plot(iterations, data['w_losses'], label='Wasserstein', color='orange', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Generator vs Discriminator vs Wasserstein Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Gradient Penalty
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(iterations, data['grad_penalties'], label='Gradient Penalty', color='green', alpha=0.7, linewidth=1.5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient Penalty')
    ax2.set_title('WGAN-GP Gradient Penalty')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Alpha (Resolution Transition)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(iterations, data['alphas'], label='Alpha', color='purple', alpha=0.7, linewidth=1.5)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Alpha')
    ax3.set_title('Resolution Transition (Alpha)')
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. All Metrics Combined
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(iterations, data['gen_losses'], label='Generator', color='blue', alpha=0.5, linewidth=2)
    ax4.plot(iterations, data['disc_losses'], label='Discriminator', color='red', alpha=0.5, linewidth=2)
    ax4.plot(iterations, data['w_losses'], label='Wasserstein', color='orange', alpha=0.5, linewidth=2)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(iterations, data['grad_penalties'], label='Grad Penalty', color='green', alpha=0.5, linewidth=2, linestyle='--')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('G & D & W Loss')
    ax4_twin.set_ylabel('Gradient Penalty')
    ax4.set_title('All Metrics Combined')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # 5. Generator Loss with Moving Average
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(iterations, data['gen_losses'], label='Generator', color='blue', alpha=0.3, linewidth=1)
    if len(iterations) > 20:
        window = min(20, len(iterations) // 10)
        g_ma = np.convolve(data['gen_losses'], np.ones(window)/window, mode='valid')
        ax5.plot(iterations[window-1:], g_ma, label=f'MA({window})', color='darkblue', linewidth=2)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Generator Loss')
    ax5.set_title('Generator Loss with Moving Average')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Discriminator Loss with Moving Average
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(iterations, data['disc_losses'], label='Discriminator', color='red', alpha=0.3, linewidth=1)
    if len(iterations) > 20:
        window = min(20, len(iterations) // 10)
        d_ma = np.convolve(data['disc_losses'], np.ones(window)/window, mode='valid')
        ax6.plot(iterations[window-1:], d_ma, label=f'MA({window})', color='darkred', linewidth=2)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Discriminator Loss')
    ax6.set_title('Discriminator Loss with Moving Average')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Resolution Steps
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(iterations, data['steps'], label='Resolution Step', color='brown', alpha=0.7, linewidth=1.5)
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Step (Resolution)')
    ax7.set_title('Training Resolution Step')
    ax7.set_yticks([1, 2, 3, 4, 5, 6])
    ax7.set_yticklabels(['8×8', '16×16', '32×32', '64×64', '128×128', '256×256'])
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Wasserstein Distance vs Gradient Penalty
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(iterations, data['w_losses'], label='Wasserstein', color='orange', alpha=0.7, linewidth=1.5)
    ax8_twin = ax8.twinx()
    ax8_twin.plot(iterations, data['grad_penalties'], label='Grad Penalty', color='green', alpha=0.7, linewidth=1.5, linestyle='--')
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('Wasserstein Distance', color='orange')
    ax8_twin.set_ylabel('Gradient Penalty', color='green')
    ax8.set_title('Wasserstein Distance vs Gradient Penalty')
    ax8.legend(loc='upper left')
    ax8_twin.legend(loc='upper right')
    ax8.grid(True, alpha=0.3)
    
    # 9. Loss Ratio (G/D)
    ax9 = fig.add_subplot(gs[2, 2])
    # Avoid division by zero
    loss_ratios = [g/abs(d) if abs(d) > 1e-8 else 0 for g, d in zip(data['gen_losses'], data['disc_losses'])]
    ax9.plot(iterations, loss_ratios, label='G/D Ratio', color='purple', alpha=0.7, linewidth=1.5)
    ax9.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Balanced (1.0)')
    ax9.set_xlabel('Iteration')
    ax9.set_ylabel('Loss Ratio (G/D)')
    ax9.set_title('Generator to Discriminator Loss Ratio')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # Overall title
    if title_prefix:
        fig.suptitle(f'{title_prefix} - Training Analysis', fontsize=16, fontweight='bold')
    else:
        fig.suptitle('Progressive GAN Training Analysis', fontsize=16, fontweight='bold')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comprehensive plot to: {output_file}")


def plot_simple(data, output_file='training_losses.png'):
    """Generate simple 2x2 training plots (matching train.py style)."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    iterations = data['iterations']
    
    # Generator, Discriminator, and Wasserstein losses
    ax1.plot(iterations, data['gen_losses'], label='Generator', color='blue', alpha=0.7)
    ax1.plot(iterations, data['disc_losses'], label='Discriminator', color='red', alpha=0.7)
    ax1.plot(iterations, data['w_losses'], label='Wasserstein', color='orange', alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Generator vs Discriminator vs Wasserstein Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gradient penalty
    ax2.plot(iterations, data['grad_penalties'], label='Gradient Penalty', color='green', alpha=0.7)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient Penalty')
    ax2.set_title('WGAN-GP Gradient Penalty')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Alpha (resolution transition)
    ax3.plot(iterations, data['alphas'], label='Alpha', color='purple', alpha=0.7)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Alpha')
    ax3.set_title('Resolution Transition (Alpha)')
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # All metrics combined
    ax4.plot(iterations, data['gen_losses'], label='Generator', color='blue', alpha=0.5, linewidth=2)
    ax4.plot(iterations, data['disc_losses'], label='Discriminator', color='red', alpha=0.5, linewidth=2)
    ax4.plot(iterations, data['w_losses'], label='Wasserstein', color='orange', alpha=0.5, linewidth=2)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(iterations, data['grad_penalties'], label='Grad Penalty', color='green', alpha=0.5, linewidth=2, linestyle='--')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('G & D & W Loss')
    ax4_twin.set_ylabel('Gradient Penalty')
    ax4.set_title('All Metrics Combined')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved simple plot to: {output_file}")


def print_statistics(data):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("TRAINING STATISTICS SUMMARY")
    print("="*60)
    
    print(f"\nTotal Iterations: {len(data['iterations'])}")
    print(f"Iteration Range: {data['iterations'][0]} to {data['iterations'][-1]}")
    
    print(f"\nGenerator Loss:")
    print(f"  Mean: {np.mean(data['gen_losses']):.6f}")
    print(f"  Std:  {np.std(data['gen_losses']):.6f}")
    print(f"  Min:  {np.min(data['gen_losses']):.6f}")
    print(f"  Max:  {np.max(data['gen_losses']):.6f}")
    
    print(f"\nDiscriminator Loss:")
    print(f"  Mean: {np.mean(data['disc_losses']):.6f}")
    print(f"  Std:  {np.std(data['disc_losses']):.6f}")
    print(f"  Min:  {np.min(data['disc_losses']):.6f}")
    print(f"  Max:  {np.max(data['disc_losses']):.6f}")
    
    print(f"\nWasserstein Distance:")
    print(f"  Mean: {np.mean(data['w_losses']):.6f}")
    print(f"  Std:  {np.std(data['w_losses']):.6f}")
    print(f"  Min:  {np.min(data['w_losses']):.6f}")
    print(f"  Max:  {np.max(data['w_losses']):.6f}")
    
    print(f"\nGradient Penalty:")
    print(f"  Mean: {np.mean(data['grad_penalties']):.6f}")
    print(f"  Std:  {np.std(data['grad_penalties']):.6f}")
    print(f"  Min:  {np.min(data['grad_penalties']):.6f}")
    print(f"  Max:  {np.max(data['grad_penalties']):.6f}")
    
    print(f"\nResolution Steps:")
    unique_steps = sorted(set(data['steps']))
    step_names = {1: '8×8', 2: '16×16', 3: '32×32', 4: '64×64', 5: '128×128', 6: '256×256'}
    print(f"  Steps trained: {[step_names.get(s, str(s)) for s in unique_steps]}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate training visualizations from statistics CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate both simple and comprehensive plots
  python plot_statistics.py training_statistics.csv
  
  # Generate only simple plot
  python plot_statistics.py training_statistics.csv --simple-only
  
  # Generate only comprehensive plot
  python plot_statistics.py training_statistics.csv --comprehensive-only
  
  # Specify output files
  python plot_statistics.py training_statistics.csv -o plots.png -c analysis.png
  
  # Add custom title
  python plot_statistics.py training_statistics.csv --title "FFHQ Trial 32"
        """
    )
    
    parser.add_argument('csv_file', type=str,
                       help='Path to training_statistics.csv file')
    parser.add_argument('-o', '--output', type=str, default='training_losses.png',
                       help='Output file for simple plot (default: training_losses.png)')
    parser.add_argument('-c', '--comprehensive', type=str, default='training_analysis.png',
                       help='Output file for comprehensive plot (default: training_analysis.png)')
    parser.add_argument('--simple-only', action='store_true',
                       help='Generate only simple plot')
    parser.add_argument('--comprehensive-only', action='store_true',
                       help='Generate only comprehensive plot')
    parser.add_argument('--title', type=str, default='',
                       help='Title prefix for plots')
    parser.add_argument('--no-stats', action='store_true',
                       help='Do not print statistics summary')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: File not found: {args.csv_file}")
        return 1
    
    # Load data
    print(f"Loading statistics from: {args.csv_file}")
    try:
        data = load_statistics(args.csv_file)
        print(f"Loaded {len(data['iterations'])} data points")
    except Exception as e:
        print(f"Error loading statistics: {e}")
        return 1
    
    # Print statistics
    if not args.no_stats:
        print_statistics(data)
    
    # Generate plots
    try:
        if args.comprehensive_only:
            plot_comprehensive(data, args.comprehensive, args.title)
        elif args.simple_only:
            plot_simple(data, args.output)
        else:
            plot_simple(data, args.output)
            plot_comprehensive(data, args.comprehensive, args.title)
        
        print("\nPlot generation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
