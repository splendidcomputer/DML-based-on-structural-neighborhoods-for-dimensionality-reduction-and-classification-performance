#!/usr/bin/env python3
"""
Convert SVG plots to PDF format for LaTeX compilation
"""
import os
import subprocess
import glob
from pathlib import Path

def convert_svg_to_pdf(svg_path, pdf_path):
    """Convert SVG to PDF using inkscape"""
    try:
        # Try inkscape first
        subprocess.run(['inkscape', '--export-type=pdf', f'--export-filename={pdf_path}', svg_path], 
                      check=True, capture_output=True)
        print(f"Converted {svg_path} -> {pdf_path}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Fallback to cairosvg if inkscape is not available
            import cairosvg
            cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
            print(f"Converted {svg_path} -> {pdf_path} using cairosvg")
            return True
        except ImportError:
            print(f"Neither inkscape nor cairosvg available. Cannot convert {svg_path}")
            return False

def main():
    # Get the script directory and find SVG files
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "python" / "results" / "plots"
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Find all SVG files
    svg_files = list(results_dir.glob("*.svg"))
    
    if not svg_files:
        print(f"No SVG files found in {results_dir}")
        return
    
    print(f"Found {len(svg_files)} SVG files to convert...")
    
    converted = 0
    for svg_file in svg_files:
        pdf_file = svg_file.with_suffix('.pdf')
        
        # Skip if PDF already exists and is newer
        if pdf_file.exists() and pdf_file.stat().st_mtime > svg_file.stat().st_mtime:
            print(f"PDF already up to date: {pdf_file}")
            converted += 1
            continue
        
        if convert_svg_to_pdf(str(svg_file), str(pdf_file)):
            converted += 1
    
    print(f"\nConversion complete: {converted}/{len(svg_files)} files converted successfully")
    
    # List the key files needed for the paper
    key_files = [
        "classifier_comparison.pdf",
        "Mean_Results_LLE_Data_Wine.pdf", 
        "Mean_Results_KernelPCA_Data_Wine.pdf",
        "Mean_Results_LLE_Data_BreastCancer.pdf",
        "Mean_Results_Isomap_Data_BreastCancer.pdf"
    ]
    
    print("\nKey files for paper:")
    for filename in key_files:
        filepath = results_dir / filename
        if filepath.exists():
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} (missing)")

if __name__ == "__main__":
    main()