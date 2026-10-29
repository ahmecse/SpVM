#!/usr/bin/env python3
"""
Visualization script for SpMV benchmark results.
Generates graphs comparing serial and parallel implementations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def plot_execution_time(df, output_dir='results'):
    """Plot execution time vs problem size."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['size'], df['serial_time_ms'], 'o-', label='Serial', 
             linewidth=2, markersize=8)
    plt.plot(df['size'], df['parallel_time_ms'], 's-', label='Parallel', 
             linewidth=2, markersize=8)
    
    plt.xlabel('Matrix Size (N×N)', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title('Execution Time vs Problem Size', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'execution_time.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

def plot_speedup(df, output_dir='results'):
    """Plot speedup vs problem size."""
    os.makedirs(output_dir, exist_ok=True)
    
    num_threads = df['num_threads'].iloc[0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['size'], df['speedup'], 'o-', label='Measured Speedup',
             linewidth=2, markersize=8, color='blue')
    plt.axhline(y=num_threads, color='red', linestyle='--',
                label=f'Ideal Speedup ({num_threads}x)', linewidth=2)
    
    plt.xlabel('Matrix Size (N×N)', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title('Speedup vs Problem Size', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'speedup.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

def plot_efficiency(df, output_dir='results'):
    """Plot parallel efficiency vs problem size."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['size'], df['efficiency'] * 100, 'o-',
             linewidth=2, markersize=8, color='green')
    plt.axhline(y=100, color='red', linestyle='--',
                label='Ideal Efficiency (100%)', linewidth=2)
    
    plt.xlabel('Matrix Size (N×N)', fontsize=12)
    plt.ylabel('Parallel Efficiency (%)', fontsize=12)
    plt.title('Parallel Efficiency vs Problem Size', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 110])
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'efficiency.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

def plot_log_scale(df, output_dir='results'):
    """Plot execution time with log scale."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(df['size'], df['serial_time_ms'], 'o-', label='Serial',
               linewidth=2, markersize=8)
    plt.loglog(df['size'], df['parallel_time_ms'], 's-', label='Parallel',
               linewidth=2, markersize=8)
    
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

def plot_all(csv_file='benchmark_results(1).csv', output_dir='results'):
    """Generate all plots from benchmark results."""
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        print("Please run the benchmark program first.")
        sys.exit(1)
    
    df = pd.read_csv(csv_file)
    
    print("\n" + "=" * 60)
    print("Generating performance graphs...")
    print("=" * 60)
    
    plot_execution_time(df, output_dir)
    plot_speedup(df, output_dir)
    plot_efficiency(df, output_dir)
    plot_log_scale(df, output_dir)
    
    print("=" * 60)
    print(f"All graphs saved to '{output_dir}/' directory")
    print("=" * 60)

if __name__ == "__main__":
    csv_file = 'benchmark_results.csv'
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    plot_all(csv_file)
