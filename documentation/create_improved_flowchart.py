#!/usr/bin/env python3
"""
Fixed BDML-MLE Flowchart Generator
Step-by-step approach to create a properly aligned flowchart
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as path_effects

def create_improved_flowchart():
    """
    Step-by-step flowchart creation with perfect arrow alignment
    """
    # Step 1: Set up the figure
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Step 2: Define color scheme
    colors = {
        'input': '#E3F2FD',     # Light blue
        'process': '#E8F5E8',   # Light green
        'decision': '#FFF3E0',  # Light orange  
        'output': '#FCE4EC',    # Light pink
        'phase': '#F3E5F5'      # Light purple
    }
    
    # Step 3: Define all boxes with precise coordinates
    boxes = [
        # Row 1: Input
        {'name': 'input', 'pos': (5, 15), 'size': (3, 0.8), 'type': 'input', 
         'text': 'Input Dataset\nX ∈ R^{n×d}'},
        
        # Row 2: Phase 1 header
        {'name': 'phase1', 'pos': (5, 13.5), 'size': (7, 0.6), 'type': 'phase',
         'text': 'Phase 1: Manifold Learning Ensemble'},
        
        # Row 3: Individual methods (horizontally aligned)
        {'name': 'pca', 'pos': (1.5, 12), 'size': (1.4, 0.6), 'type': 'process', 'text': 'PCA'},
        {'name': 'lda', 'pos': (3.2, 12), 'size': (1.4, 0.6), 'type': 'process', 'text': 'LDA'},
        {'name': 'lle', 'pos': (4.9, 12), 'size': (1.4, 0.6), 'type': 'process', 'text': 'LLE'},
        {'name': 'isomap', 'pos': (6.6, 12), 'size': (1.4, 0.6), 'type': 'process', 'text': 'Isomap'},
        {'name': 'others', 'pos': (8.3, 12), 'size': (1.4, 0.6), 'type': 'process', 'text': 'MDS\nKPCA'},
        
        # Row 4: Embedding result
        {'name': 'embedding', 'pos': (5, 10.5), 'size': (3.5, 0.8), 'type': 'process',
         'text': 'Low-Dimensional\nEmbedding Y ∈ R^{n×k}'},
        
        # Row 5: Phase 2 header  
        {'name': 'phase2', 'pos': (5, 9), 'size': (7, 0.6), 'type': 'phase',
         'text': 'Phase 2: Balanced Neighborhood Construction'},
        
        # Row 6: Neighborhood construction (horizontally aligned)
        {'name': 'knn', 'pos': (2.5, 7.5), 'size': (2.2, 0.8), 'type': 'process',
         'text': 'k-NN Graph\nConstruction'},
        {'name': 'similar', 'pos': (5, 7.5), 'size': (2.2, 0.8), 'type': 'process', 
         'text': 'Similar Set S_i\n{same class}'},
        {'name': 'dissimilar', 'pos': (7.5, 7.5), 'size': (2.2, 0.8), 'type': 'process',
         'text': 'Dissimilar Set D_i\n{different class}'},
        
        # Row 7: Balance constraint
        {'name': 'balance', 'pos': (5, 6), 'size': (4, 0.8), 'type': 'decision',
         'text': 'Balance Constraint:\n|S_i| = |D_i| = k'},
        
        # Row 8: Metric learning
        {'name': 'metric', 'pos': (5, 4.5), 'size': (5, 0.8), 'type': 'process',
         'text': 'Distance Metric Learning\nmin Σ d_M²(similar) - α Σ d_M²(dissimilar)'},
        
        # Row 9: Convergence check
        {'name': 'converge', 'pos': (5, 3), 'size': (2.5, 0.8), 'type': 'decision',
         'text': 'Converged?'},
        
        # Row 10: Output
        {'name': 'output', 'pos': (5, 1.5), 'size': (3, 0.8), 'type': 'output',
         'text': 'Learned Metric M'}
    ]
    
    # Step 4: Draw boxes and store their actual boundaries
    box_info = {}
    for box in boxes:
        x, y = box['pos']
        w, h = box['size']
        
        # Create box
        rect = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.05",
            facecolor=colors[box['type']],
            edgecolor='black',
            linewidth=1.2,
            alpha=0.9
        )
        ax.add_patch(rect)
        
        # Store box boundaries for precise arrow positioning
        box_info[box['name']] = {
            'center': (x, y),
            'left': x - w/2,
            'right': x + w/2, 
            'top': y + h/2,
            'bottom': y - h/2,
            'width': w,
            'height': h
        }
        
        # Add text with better formatting
        text = ax.text(x, y, box['text'], ha='center', va='center', 
                      fontsize=9, weight='bold', 
                      bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
    
    # Step 5: Draw arrows with precise positioning
    def draw_precise_arrow(from_box, to_box, offset_from=(0, 0), offset_to=(0, 0), 
                          color='#2E7D32', style='->', linestyle='-', 
                          connectionstyle='arc3,rad=0'):
        """Draw arrow with precise box edge connections"""
        from_info = box_info[from_box]
        to_info = box_info[to_box]
        
        # Calculate connection points at box edges
        if from_info['center'][1] > to_info['center'][1]:  # Going down
            start_x = from_info['center'][0] + offset_from[0]
            start_y = from_info['bottom'] + offset_from[1]
            end_x = to_info['center'][0] + offset_to[0]
            end_y = to_info['top'] + offset_to[1]
        else:  # Going up
            start_x = from_info['center'][0] + offset_from[0]
            start_y = from_info['top'] + offset_from[1]
            end_x = to_info['center'][0] + offset_to[0] 
            end_y = to_info['bottom'] + offset_to[1]
        
        arrow = FancyArrowPatch(
            (start_x, start_y), (end_x, end_y),
            connectionstyle=connectionstyle,
            arrowstyle=style,
            color=color,
            linewidth=2,
            linestyle=linestyle,
            alpha=0.8
        )
        ax.add_patch(arrow)
        return arrow
    
    # Step 6: Draw all arrows systematically
    
    # Input to Phase 1
    draw_precise_arrow('input', 'phase1')
    
    # Phase 1 to individual methods
    methods = ['pca', 'lda', 'lle', 'isomap', 'others']
    for method in methods:
        # From phase1 center to each method top
        method_info = box_info[method]
        phase1_info = box_info['phase1']
        
        arrow = FancyArrowPatch(
            (phase1_info['center'][0], phase1_info['bottom']),
            (method_info['center'][0], method_info['top']),
            arrowstyle='->',
            color='#1976D2',
            linewidth=1.5,
            alpha=0.7
        )
        ax.add_patch(arrow)
    
    # Methods to embedding (convergence arrows)
    for method in methods:
        method_info = box_info[method]
        embed_info = box_info['embedding']
        
        arrow = FancyArrowPatch(
            (method_info['center'][0], method_info['bottom']),
            (embed_info['center'][0], embed_info['top']),
            arrowstyle='->',
            color='#388E3C',
            linewidth=1.5,
            alpha=0.7
        )
        ax.add_patch(arrow)
    
    # Embedding to Phase 2
    draw_precise_arrow('embedding', 'phase2')
    
    # Phase 2 to neighborhood construction
    neighborhood_boxes = ['knn', 'similar', 'dissimilar'] 
    for box_name in neighborhood_boxes:
        phase2_info = box_info['phase2']
        box_info_current = box_info[box_name]
        
        arrow = FancyArrowPatch(
            (phase2_info['center'][0], phase2_info['bottom']),
            (box_info_current['center'][0], box_info_current['top']),
            arrowstyle='->',
            color='#F57C00',
            linewidth=1.5,
            alpha=0.7
        )
        ax.add_patch(arrow)
    
    # Neighborhood boxes to balance
    for box_name in neighborhood_boxes:
        draw_precise_arrow(box_name, 'balance')
    
    # Balance to metric learning
    draw_precise_arrow('balance', 'metric')
    
    # Metric to convergence
    draw_precise_arrow('metric', 'converge')
    
    # Convergence YES to output
    converge_info = box_info['converge']
    output_info = box_info['output']
    
    yes_arrow = FancyArrowPatch(
        (converge_info['center'][0], converge_info['bottom']),
        (output_info['center'][0], output_info['top']),
        arrowstyle='->',
        color='green',
        linewidth=2.5,
        alpha=0.9
    )
    ax.add_patch(yes_arrow)
    
    # Add YES label
    ax.text(converge_info['center'][0] + 0.3, 
           (converge_info['bottom'] + output_info['top'])/2,
           'YES', fontsize=10, weight='bold', color='green',
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.8))
    
    # Convergence NO back to metric (curved arrow)
    no_arrow = FancyArrowPatch(
        (converge_info['right'], converge_info['center'][1]),
        (box_info['metric']['right'], box_info['metric']['center'][1]),
        connectionstyle="arc3,rad=0.3",
        arrowstyle='->',
        color='red',
        linewidth=2,
        linestyle='--',
        alpha=0.8
    )
    ax.add_patch(no_arrow)
    
    # Add NO label
    ax.text(converge_info['center'][0] + 1.8, converge_info['center'][1] + 0.5,
           'NO', fontsize=10, weight='bold', color='red',
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral', alpha=0.8))
    
    # Step 7: Add title and labels
    ax.text(5, 15.8, 'BDML-MLE Algorithm Flowchart', 
           ha='center', va='center', fontsize=18, weight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    ax.text(5, 15.4, 'Balanced Distance Metric Learning with Manifold Learning Ensemble', 
           ha='center', va='center', fontsize=12, style='italic')
    
    # Step 8: Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['input'], edgecolor='black', label='Input/Output'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['phase'], edgecolor='black', label='Phase Header'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['process'], edgecolor='black', label='Process'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['decision'], edgecolor='black', label='Decision')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Step 9: Save the improved flowchart
    plt.tight_layout()
    plt.savefig('/home/mostafa/programming/myprograms/DML-based-on-structural-neighborhoods-for-dimensionality-reduction-and-classification-performance/documentation/bdml_mle_improved_flowchart.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Improved BDML-MLE flowchart created successfully!")
    print("📁 File: bdml_mle_improved_flowchart.pdf")
    print("🎯 Features: Precise arrow alignment, clear visual hierarchy, proper spacing")

if __name__ == "__main__":
    create_improved_flowchart()