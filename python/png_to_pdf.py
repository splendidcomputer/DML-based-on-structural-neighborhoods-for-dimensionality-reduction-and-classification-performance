#!/usr/bin/env python3
"""
Simple PNG to PDF converter for academic publication.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import glob
from pathlib import Path

def convert_png_to_pdf():
    """Convert existing PNG plots to PDF format."""
    
    plots_dir = Path('results/plots')
    png_files = list(plots_dir.glob('*.png'))
    
    print(f"Found {len(png_files)} PNG files to convert")
    
    for png_file in png_files:
        # Create PDF filename
        pdf_file = png_file.with_suffix('.pdf')
        
        try:
            # Load the PNG image
            img = Image.open(png_file)
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(img)
            ax.axis('off')
            
            # Save as PDF
            plt.savefig(pdf_file, format='pdf', bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Converted: {png_file.name} -> {pdf_file.name}")
            
        except Exception as e:
            print(f"Error converting {png_file.name}: {e}")

if __name__ == "__main__":
    print("Converting PNG plots to PDF format...")
    convert_png_to_pdf()
    print("Conversion complete!")