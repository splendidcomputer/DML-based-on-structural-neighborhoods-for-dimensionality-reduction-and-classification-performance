#!/usr/bin/env python3
"""
Generate SVG vector graphics plots from existing results for academic publication.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set publication-quality parameters
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'svg',
    'text.usetex': False  # Set to True if LaTeX is available
})

def generate_svg_plots():
    """Generate SVG plots from existing CSV results."""
    
    results_dir = Path('results/comprehensive')
    plots_dir = Path('results/plots')
    plots_dir.mkdir(exist_ok=True)
    
    datasets = ['Iris', 'Wine', 'BreastCancer', 'Vehicle']
    dr_methods = ['PCA', 'LDA', 'MDS', 'Isomap', 'LLE', 'KernelPCA', 'Autoencoder']
    
    for dataset_name in datasets:
        print(f"Creating SVG plots for {dataset_name}...")
        
        for dr_method in dr_methods:
            # Try to load results for this combination
            result_files = {
                'KNN': results_dir / f"{dataset_name}_{dr_method}_KNN_results.csv",
                'simKNN': results_dir / f"{dataset_name}_{dr_method}_simKNN_results.csv", 
                'SVM': results_dir / f"{dataset_name}_{dr_method}_SVM_results.csv"
            }
            
            # Check if files exist
            if not all(f.exists() for f in result_files.values()):
                # Try alternative naming pattern
                result_files = {
                    'KNN': results_dir / f"{dataset_name}_KNN_{dr_method}.csv",
                    'simKNN': results_dir / f"{dataset_name}_simKNN_{dr_method}.csv", 
                    'SVM': results_dir / f"{dataset_name}_SVM_{dr_method}.csv"
                }
                
                if not all(f.exists() for f in result_files.values()):
                    continue
                
            # Load data
            data = {}
            for classifier, filepath in result_files.items():
                try:
                    df = pd.read_csv(filepath)
                    if len(df) > 0:
                        data[classifier] = df
                except Exception as e:
                    print(f"  Error loading {filepath}: {e}")
                    continue
            
            if not data:
                continue
                
            # Create publication-quality plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
            
            # Colors and styles for classifiers (academic style)
            colors = {'KNN': '#1f77b4', 'simKNN': '#d62728', 'SVM': '#2ca02c'}
            markers = {'KNN': 'o', 'simKNN': 's', 'SVM': '^'}
            linestyles = {'KNN': '-', 'simKNN': '--', 'SVM': '-.'}
            
            for classifier, df in data.items():
                if len(df) == 0:
                    continue
                    
                x = df['Dimension'].values if 'Dimension' in df.columns else np.arange(len(df))
                
                # Plot 1: Sensitivity
                if 'Sensitivity' in df.columns:
                    y = df['Sensitivity'].values * 100
                    ax1.plot(x, y, marker=markers[classifier], 
                           color=colors[classifier], label=classifier, 
                           linewidth=2, markersize=5, linestyle=linestyles[classifier])
                
                # Plot 2: Specificity  
                if 'Specificity' in df.columns:
                    y = df['Specificity'].values * 100
                    ax2.plot(x, y, marker=markers[classifier],
                           color=colors[classifier], label=classifier, 
                           linewidth=2, markersize=5, linestyle=linestyles[classifier])
                
                # Plot 3: Accuracy
                if 'Accuracy' in df.columns:
                    y = df['Accuracy'].values * 100
                    ax3.plot(x, y, marker=markers[classifier],
                           color=colors[classifier], label=classifier, 
                           linewidth=2, markersize=5, linestyle=linestyles[classifier])
                
                # Plot 4: Time
                if 'Time' in df.columns:
                    y = df['Time'].values
                    ax4.plot(x, y, marker=markers[classifier],
                           color=colors[classifier], label=classifier, 
                           linewidth=2, markersize=5, linestyle=linestyles[classifier])
            
            # Formatting with publication standards
            def format_subplot(ax, ylabel, title, ylim_range=None):
                ax.set_xlabel('Dimensionality of the embedded space', fontweight='bold')
                ax.set_ylabel(ylabel, fontweight='bold')
                ax.set_title(title, fontweight='bold', fontsize=13)
                ax.legend(frameon=True, fancybox=True, shadow=True)
                ax.grid(True, alpha=0.3, linestyle=':')
                if ylim_range:
                    ax.set_ylim(ylim_range)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            format_subplot(ax1, 'Mean sensitivity (%)', f'Mean sensitivity using {dr_method}', (0, 100))
            format_subplot(ax2, 'Mean specificity (%)', f'Mean specificity using {dr_method}', (0, 100))
            format_subplot(ax3, 'Mean accuracy (%)', f'Mean accuracy using {dr_method}', (0, 100))
            format_subplot(ax4, 'Mean elapsed time (s)', f'Mean elapsed time using {dr_method}')
            
            plt.tight_layout()
            
            # Save as SVG (vector graphics)
            plot_filename = f"Mean_Results_{dr_method}_Data_{dataset_name}.svg"
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, format='svg', bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"  Created: {plot_filename}")

if __name__ == "__main__":
    print("Generating SVG vector graphics plots...")
    generate_svg_plots()
    print("SVG generation complete!")