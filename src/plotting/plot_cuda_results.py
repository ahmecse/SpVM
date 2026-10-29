#!/usr/bin/env python3
"""
Visualization script for CUDA SpMV benchmark results.
Generates graphs comparing serial, OpenMP, and CUDA implementations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def plot_execution_time(df, output_dir='results_cuda'):
    """Plot execution time vs problem size for all implementations."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['size'], df['serial_time_ms'], 'o-', label='Serial', 
             linewidth=2, markersize=8, color='blue')
    plt.plot(df['size'], df['openmp_time_ms'], 's-', label='OpenMP', 
             linewidth=2, markersize=8, color='green')
    plt.plot(df['size'], df['cuda_time_ms'], '^-', label='CUDA', 
             linewidth=2, markersize=8, color='red')
    
    plt.xlabel('Matrix Size (N×N)', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title('Execution Time Comparison: Serial vs OpenMP vs CUDA', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'execution_time_cuda.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

def plot_speedup_comparison(df, output_dir='results_cuda'):
    """Plot speedup comparison between OpenMP and CUDA."""
    os.makedirs(output_dir, exist_ok=True)
    
    num_threads = df['num_threads'].iloc[0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['size'], df['openmp_speedup'], 's-', label='OpenMP Speedup',
             linewidth=2, markersize=8, color='green')
    plt.plot(df['size'], df['cuda_speedup'], '^-', label='CUDA Speedup',
             linewidth=2, markersize=8, color='red')
    plt.axhline(y=num_threads, color='gray', linestyle='--',
                label=f'OpenMP Ideal ({num_threads}x)', linewidth=2, alpha=0.7)
    
    plt.xlabel('Matrix Size (N×N)', fontsize=12)
    plt.ylabel('Speedup vs Serial', fontsize=12)
    plt.title('Speedup Comparison: OpenMP vs CUDA', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'speedup_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

def plot_performance_bars(df, output_dir='results_cuda'):
    """Plot bar chart comparing performance across implementations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select a few representative sizes
    if len(df) > 6:
        indices = [0, len(df)//3, 2*len(df)//3, len(df)-1]
        subset = df.iloc[indices]
    else:
        subset = df
    
    x = np.arange(len(subset))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, subset['serial_time_ms'], width, 
                   label='Serial', color='blue', alpha=0.8)
    bars2 = ax.bar(x, subset['openmp_time_ms'], width, 
                   label='OpenMP', color='green', alpha=0.8)
    bars3 = ax.bar(x + width, subset['cuda_time_ms'], width, 
                   label='CUDA', color='red', alpha=0.8)
    
    ax.set_xlabel('Matrix Size', fontsize=12)
    ax.set_ylabel('Execution Time (ms)', fontsize=12)
    ax.set_title('Performance Comparison Across Problem Sizes', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}×{s}" for s in subset['size']])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'performance_bars.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

def plot_log_scale(df, output_dir='results_cuda'):
    """Plot execution time with log scale."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(df['size'], df['serial_time_ms'], 'o-', label='Serial',
               linewidth=2, markersize=8, color='blue')
    plt.loglog(df['size'], df['openmp_time_ms'], 's-', label='OpenMP',
               linewidth=2, markersize=8, color='green')
    plt.loglog(df['size'], df['cuda_time_ms'], '^-', label='CUDA',
               linewidth=2, markersize=8, color='red')
    
    plt.xlabel('Matrix Size (N×N)', fontsize=12)
    plt.ylabel('Execution Time (ms, log scale)', fontsize=12)
    plt.title('Execution Time Comparison (Log-Log Scale)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'execution_time_log.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

def plot_relative_performance(df, output_dir='results_cuda'):
    """Plot relative performance (CUDA vs OpenMP)."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate how many times faster CUDA is than OpenMP
    cuda_vs_openmp = df['openmp_time_ms'] / df['cuda_time_ms']
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['size'], cuda_vs_openmp, 'o-', 
             linewidth=2, markersize=8, color='purple')
    plt.axhline(y=1.0, color='red', linestyle='--', 
                label='Equal Performance', linewidth=2)
    
    plt.xlabel('Matrix Size (N×N)', fontsize=12)
    plt.ylabel('CUDA Performance / OpenMP Performance', fontsize=12)
    plt.title('CUDA vs OpenMP Relative Performance', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'cuda_vs_openmp.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

def plot_all(csv_file='benchmark_cuda_results.csv', output_dir='results_cuda'):
    """Generate all plots from benchmark results."""
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        print("Please run the CUDA benchmark program first.")
        sys.exit(1)
    
    df = pd.read_csv(csv_file)
    
    print("\n" + "=" * 60)
    print("Generating CUDA performance graphs...")
    print("=" * 60)
    
    plot_execution_time(df, output_dir)
    plot_speedup_comparison(df, output_dir)
    plot_performance_bars(df, output_dir)
    plot_log_scale(df, output_dir)
    plot_relative_performance(df, output_dir)
    
    print("=" * 60)
    print(f"All graphs saved to '{output_dir}/' directory")
    print("=" * 60)
    
    print("\nPerformance Summary:")
    print(f"  Average OpenMP Speedup: {df['openmp_speedup'].mean():.2f}x")
    print(f"  Average CUDA Speedup:   {df['cuda_speedup'].mean():.2f}x")
    print(f"  Maximum CUDA Speedup:   {df['cuda_speedup'].max():.2f}x")
    
    cuda_vs_openmp = df['openmp_time_ms'] / df['cuda_time_ms']
    print(f"\nCUDA vs OpenMP:")
    print(f"  Average: {cuda_vs_openmp.mean():.2f}x faster")
    print(f"  Maximum: {cuda_vs_openmp.max():.2f}x faster")

if __name__ == "__main__":
    csv_file = 'benchmark_cuda_results.csv'
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    plot_all(csv_file)
