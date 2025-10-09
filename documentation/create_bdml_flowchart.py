#!/usr/bin/env python3
"""
BDML-MLE Flowchart Generator
Creates a well-aligned flowchart with proper arrow positioning
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch

# Set high-quality output settings
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 14,
    'font.family': 'serif',
    'text.usetex': False
})

def create_bdml_mle_flowchart():
    """Create a well-designed BDML-MLE algorithm flowchart with proper alignment"""
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define colors
    colors = {
        'input': '#E8F4FD',      # Light blue
        'process': '#B8E6B8',    # Light green 
        'decision': '#FFE4B5',   # Light orange
        'output': '#FFB6C1',     # Light pink
        'phase': '#D8BFD8',      # Light purple
        'arrow': '#4A90E2'       # Blue
    }
    
    # Step 1: Define all box positions and properties with better alignment
    boxes = [
        # Input
        {'id': 'input', 'pos': (6, 13), 'size': (3, 0.8), 'text': 'High-Dimensional\nDataset X ∈ ℝⁿˣᵈ', 'type': 'input'},
        
        # Phase 1 Header
        {'id': 'phase1', 'pos': (6, 11.5), 'size': (8, 0.6), 'text': 'Phase 1: Manifold Learning Ensemble', 'type': 'phase'},
        
        # Manifold learning methods (arranged horizontally)
        {'id': 'pca', 'pos': (2, 10.2), 'size': (1.8, 0.6), 'text': 'PCA', 'type': 'process'},
        {'id': 'lda', 'pos': (4.2, 10.2), 'size': (1.8, 0.6), 'text': 'LDA', 'type': 'process'},
        {'id': 'lle', 'pos': (6.4, 10.2), 'size': (1.8, 0.6), 'text': 'LLE', 'type': 'process'},
        {'id': 'isomap', 'pos': (8.6, 10.2), 'size': (1.8, 0.6), 'text': 'Isomap', 'type': 'process'},
        {'id': 'others', 'pos': (10.8, 10.2), 'size': (1.8, 0.6), 'text': 'MDS/\nKPCA', 'type': 'process'},
        
        # Embedding result
        {'id': 'embedding', 'pos': (6, 8.8), 'size': (3.5, 0.8), 'text': 'Low-Dimensional\nEmbedding Y ∈ ℝⁿˣᵏ', 'type': 'process'},
        
        # Phase 2 Header
        {'id': 'phase2', 'pos': (6, 7.3), 'size': (8, 0.6), 'text': 'Phase 2: Balanced Neighborhood Construction', 'type': 'phase'},
        
        # Neighborhood construction steps
        {'id': 'knn', 'pos': (3, 6), 'size': (2.5, 0.8), 'text': 'k-NN Graph\nConstruction', 'type': 'process'},
        {'id': 'similar', 'pos': (6, 6), 'size': (2.5, 0.8), 'text': 'Similar Set\nS_i = {same class}', 'type': 'process'},
        {'id': 'dissimilar', 'pos': (9, 6), 'size': (2.5, 0.8), 'text': 'Dissimilar Set\nD_i = {diff class}', 'type': 'process'},
        
        # Balancing step
        {'id': 'balance', 'pos': (6, 4.5), 'size': (4, 0.8), 'text': 'Balance: |S_i| = |D_i| = k\n(Address Class Imbalance)', 'type': 'decision'},
        
        # Metric learning
        {'id': 'metric', 'pos': (6, 3), 'size': (5, 0.8), 'text': 'Distance Metric Learning\nmin Σ d_M²(similar) - α Σ d_M²(dissimilar)', 'type': 'process'},
        
        # Convergence check
        {'id': 'converge', 'pos': (6, 1.5), 'size': (2.5, 0.8), 'text': 'Converged?', 'type': 'decision'},
        
        # Output
        {'id': 'output', 'pos': (6, 0.3), 'size': (3, 0.6), 'text': 'Learned Metric M', 'type': 'output'}
    ]
    
    # Step 2: Draw all boxes
    box_objects = {}
    for box in boxes:
        x, y = box['pos']
        w, h = box['size']
        
        # Create rounded rectangle
        rect = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.05",
            facecolor=colors[box['type']],
            edgecolor='black',
            linewidth=1.2
        )
        
        ax.add_patch(rect)
        box_objects[box['id']] = {'pos': (x, y), 'size': (w, h)}
        
        # Add text
        ax.text(x, y, box['text'], ha='center', va='center', 
               fontsize=10, weight='bold', wrap=True)
    
    # Step 3: Define arrow connections with precise positioning
    def draw_arrow(start_box, end_box, offset_start=(0, 0), offset_end=(0, 0), 
                   color='#4A90E2', style='->', linestyle='-', curved=False):
        """Draw arrow between two boxes with precise positioning"""
        start_pos = box_objects[start_box]['pos']
        end_pos = box_objects[end_box]['pos']
        start_size = box_objects[start_box]['size']
        end_size = box_objects[end_box]['size']
        
        # Calculate connection points at box edges
        start_x = start_pos[0] + offset_start[0]
        start_y = start_pos[1] - start_size[1]/2 + offset_start[1]
        
        end_x = end_pos[0] + offset_end[0]  
        end_y = end_pos[1] + end_size[1]/2 + offset_end[1]
        
        if curved:
            # Create curved arrow
            arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                  connectionstyle="arc3,rad=0.3",
                                  arrowstyle=style,
                                  color=color,
                                  linewidth=2,
                                  linestyle=linestyle)
        else:
            # Create straight arrow
            arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                  arrowstyle=style,
                                  color=color,
                                  linewidth=2,
                                  linestyle=linestyle)
        
        ax.add_patch(arrow)
    
    # Step 4: Draw all arrows with proper positioning
    
    # Input to Phase 1
    draw_arrow('input', 'phase1', offset_start=(0, 0), offset_end=(0, 0))
    
    # Phase 1 to individual methods
    for method in ['pca', 'lda', 'lle', 'isomap', 'others']:
        # From phase1 to each method
        start_x = 6  # Phase 1 center
        start_y = 11.5 - 0.3  # Bottom of phase 1
        method_pos = box_objects[method]['pos']
        end_x = method_pos[0]
        end_y = method_pos[1] + 0.3  # Top of method box
        
        arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                              arrowstyle='->',
                              color=colors['arrow'],
                              linewidth=1.5)
        ax.add_patch(arrow)
    
    # Methods to embedding (convergence)
    for method in ['pca', 'lda', 'lle', 'isomap', 'others']:
        method_pos = box_objects[method]['pos']
        start_x = method_pos[0]
        start_y = method_pos[1] - 0.3  # Bottom of method
        end_x = 6  # Embedding center
        end_y = 8.8 + 0.4  # Top of embedding
        
        arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                              arrowstyle='->',
                              color=colors['arrow'],
                              linewidth=1.5)
        ax.add_patch(arrow)
    
    # Embedding to Phase 2
    draw_arrow('embedding', 'phase2', offset_start=(0, 0), offset_end=(0, 0))
    
    # Phase 2 to neighborhood construction
    phase2_bottom = 7.3 - 0.3
    for box_id, x_pos in [('knn', 3), ('similar', 6), ('dissimilar', 9)]:
        arrow = FancyArrowPatch((6, phase2_bottom), (x_pos, 6.4),
                              arrowstyle='->',
                              color=colors['arrow'],
                              linewidth=1.5)
        ax.add_patch(arrow)
    
    # Neighborhood boxes to balance
    for box_id in ['knn', 'similar', 'dissimilar']:
        draw_arrow(box_id, 'balance', offset_start=(0, 0), offset_end=(0, 0))
    
    # Balance to metric learning
    draw_arrow('balance', 'metric', offset_start=(0, 0), offset_end=(0, 0))
    
    # Metric to convergence
    draw_arrow('metric', 'converge', offset_start=(0, 0), offset_end=(0, 0))
    
    # Convergence YES to output
    conv_pos = box_objects['converge']['pos']
    output_pos = box_objects['output']['pos']
    
    arrow = FancyArrowPatch((conv_pos[0], conv_pos[1] - 0.4), 
                          (output_pos[0], output_pos[1] + 0.3),
                          arrowstyle='->',
                          color='green',
                          linewidth=2)
    ax.add_patch(arrow)
    ax.text(conv_pos[0] + 0.3, conv_pos[1] - 0.2, 'YES', fontsize=10, 
           color='green', weight='bold')
    
    # Convergence NO back to metric (curved)
    arrow = FancyArrowPatch((conv_pos[0] + 1.25, conv_pos[1]), 
                          (box_objects['metric']['pos'][0] + 2.5, 
                           box_objects['metric']['pos'][1]),
                          connectionstyle="arc3,rad=0.5",
                          arrowstyle='->',
                          color='red',
                          linewidth=2,
                          linestyle='--')
    ax.add_patch(arrow)
    ax.text(conv_pos[0] + 2, conv_pos[1] + 0.5, 'NO', fontsize=10, 
           color='red', weight='bold')
    
    # Add title
    ax.text(6, 13.8, 'BDML-MLE Algorithm Flowchart', 
           ha='center', va='center', fontsize=16, weight='bold')
    ax.text(6, 13.5, 'Balanced Distance Metric Learning with Manifold Learning Ensemble', 
           ha='center', va='center', fontsize=12, style='italic')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['input'], edgecolor='black', label='Input/Output'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['phase'], edgecolor='black', label='Phase Header'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['process'], edgecolor='black', label='Process'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['decision'], edgecolor='black', label='Decision/Balance')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig('/home/mostafa/programming/myprograms/DML-based-on-structural-neighborhoods-for-dimensionality-reduction-and-classification-performance/documentation/bdml_mle_flowchart.pdf', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("BDML-MLE flowchart created successfully!")

if __name__ == "__main__":
    create_bdml_mle_flowchart()