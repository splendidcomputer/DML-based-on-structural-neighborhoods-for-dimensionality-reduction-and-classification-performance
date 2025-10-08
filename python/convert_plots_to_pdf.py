#!/usr/bin/env python3
"""
Convert existing PNG plots to PDF format for academic publication.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results_and_create_pdf_plots():
    """Load existing results and create PDF plots."""
    
    results_dir = Path('results/comprehensive')
    plots_dir = Path('results/plots')
    plots_dir.mkdir(exist_ok=True)
    
    datasets = ['Iris', 'Wine', 'BreastCancer', 'Vehicle']
    dr_methods = ['PCA', 'LDA', 'MDS', 'Isomap', 'LLE', 'KernelPCA', 'Autoencoder']
    
    for dataset_name in datasets:
        print(f"Creating PDF plots for {dataset_name}...")
        
        for dr_method in dr_methods:
            # Try to load results for this combination
            result_files = {
                'KNN': results_dir / f"{dataset_name}_{dr_method}_KNN_results.csv",
                'simKNN': results_dir / f"{dataset_name}_{dr_method}_simKNN_results.csv", 
                'SVM': results_dir / f"{dataset_name}_{dr_method}_SVM_results.csv"
            }
            
            # Check if files exist
            if not all(f.exists() for f in result_files.values()):
                continue
                
            # Load data
            data = {}
            for classifier, filepath in result_files.items():
                try:
                    df = pd.read_csv(filepath)
                    data[classifier] = df
                except:
                    continue
            
            if not data:
                continue
                
            # Create MATLAB-style plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Colors for classifiers
            colors = {'KNN': 'blue', 'simKNN': 'red', 'SVM': 'green'}
            markers = {'KNN': 'o-', 'simKNN': 's-', 'SVM': '^-'}
            
            for classifier, df in data.items():
                if len(df) == 0:
                    continue
                    
                x = df['Dimension'].values
                
                # Plot 1: Sensitivity
                if 'Sensitivity' in df.columns:
                    ax1.plot(x, df['Sensitivity']*100, markers[classifier], 
                           color=colors[classifier], label=classifier, linewidth=2, markersize=6)
                
                # Plot 2: Specificity  
                if 'Specificity' in df.columns:
                    ax2.plot(x, df['Specificity']*100, markers[classifier],
                           color=colors[classifier], label=classifier, linewidth=2, markersize=6)
                
                # Plot 3: Accuracy
                if 'Accuracy' in df.columns:
                    ax3.plot(x, df['Accuracy']*100, markers[classifier],
                           color=colors[classifier], label=classifier, linewidth=2, markersize=6)
                
                # Plot 4: Time
                if 'Time' in df.columns:
                    ax4.plot(x, df['Time'], markers[classifier],
                           color=colors[classifier], label=classifier, linewidth=2, markersize=6)
            
            # Formatting
            ax1.set_xlabel('Dimensionality of the embedded space')
            ax1.set_ylabel('Mean sensitivity (%)')
            ax1.set_title(f'Mean sensitivity using {dr_method}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
            
            ax2.set_xlabel('Dimensionality of the embedded space')
            ax2.set_ylabel('Mean specificity (%)')
            ax2.set_title(f'Mean specificity using {dr_method}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)
            
            ax3.set_xlabel('Dimensionality of the embedded space')
            ax3.set_ylabel('Mean accuracy (%)')
            ax3.set_title(f'Mean accuracy using {dr_method}')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 100)
            
            ax4.set_xlabel('Dimensionality of the embedded space')
            ax4.set_ylabel('Mean elapsed time (s)')
            ax4.set_title(f'Mean elapsed time using {dr_method}')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save as PDF
            plot_filename = f"Mean_Results_{dr_method}_Data_{dataset_name}.pdf"
            plot_path = plots_dir / plot_filename
            plt.savefig(plot_path, format='pdf', bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"  Created: {plot_filename}")

if __name__ == "__main__":
    print("Converting plots to PDF format...")
    load_results_and_create_pdf_plots()
    print("PDF conversion complete!")