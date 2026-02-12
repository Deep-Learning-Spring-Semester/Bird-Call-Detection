"""
Generate DEEPNET architecture diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

# Color scheme
color_input = '#E8F4F8'
color_stem = '#B8E6F0'
color_freq = '#FFB6C1'
color_temp = '#87CEEB'
color_fusion = '#DDA0DD'
color_pool = '#98FB98'
color_classifier = '#FFD700'

def draw_block(ax, x, y, width, height, text, color, fontsize=10, fontweight='normal'):
    """Draw a rectangular block with text"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.05",
                         edgecolor='black',
                         facecolor=color,
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center',
           fontsize=fontsize, fontweight=fontweight,
           wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, style='->'):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=style,
                          color='black',
                          linewidth=2,
                          mutation_scale=20)
    ax.add_patch(arrow)

# Title
ax.text(5, 19, 'DEEPNET Architecture',
       ha='center', fontsize=18, fontweight='bold')
ax.text(5, 18.5, '~2.1M Parameters | 18-Class Bird Call Classifier',
       ha='center', fontsize=11, style='italic')

# Input
y_pos = 17.5
draw_block(ax, 2, y_pos, 6, 0.6, 'Input: Mel-Spectrogram\n(1 × 128 × 216)', color_input, fontsize=11, fontweight='bold')
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)

# StemBlock
y_pos = 16.3
draw_block(ax, 2, y_pos, 6, 0.8, 'StemBlock\nConv2D (3×3, 32) + ReLU + BN\nConv2D (3×3, 32) + ReLU + BN',
          color_stem, fontsize=9)
ax.text(8.5, y_pos + 0.4, 'Initial feature\nextraction', fontsize=8, style='italic', color='gray')
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)

# DualPathBlock 1
y_pos = 14.5
ax.text(5, y_pos + 0.8, 'DualPathBlock 1 (32 → 64 channels)',
       ha='center', fontsize=11, fontweight='bold')

# Split arrow
draw_arrow(ax, 5, y_pos + 0.6, 3, y_pos + 0.3)
draw_arrow(ax, 5, y_pos + 0.6, 7, y_pos + 0.3)

# Frequency branch
draw_block(ax, 1, y_pos - 0.5, 3, 0.7, 'Frequency Branch\nConv2D (5×1, 64)\nCaptures pitch,\nharmonics',
          color_freq, fontsize=8)

# Temporal branch
draw_block(ax, 6, y_pos - 0.5, 3, 0.7, 'Temporal Branch\nConv2D (1×5, 64)\nCaptures rhythm,\nduration',
          color_temp, fontsize=8)

# Merge arrows
draw_arrow(ax, 2.5, y_pos - 0.5, 4.5, y_pos - 1.3)
draw_arrow(ax, 7.5, y_pos - 0.5, 5.5, y_pos - 1.3)

# Fusion
y_pos = 12.8
draw_block(ax, 2.5, y_pos, 5, 0.6, 'Fusion Layer\nConcatenate + Conv2D (1×1, 64)',
          color_fusion, fontsize=9)

# Downsample + Residual
y_pos = 11.8
draw_block(ax, 2.5, y_pos, 5, 0.5, 'Downsample (MaxPool 2×2) + Projection Shortcut',
          color_fusion, fontsize=8)
ax.text(8.5, y_pos + 0.25, 'Output:\n64×54', fontsize=7, style='italic', color='gray')
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)

# DualPathBlock 2
y_pos = 10.3
ax.text(5, y_pos + 0.8, 'DualPathBlock 2 (64 → 256 channels)',
       ha='center', fontsize=11, fontweight='bold')

# Split arrow
draw_arrow(ax, 5, y_pos + 0.6, 3, y_pos + 0.3)
draw_arrow(ax, 5, y_pos + 0.6, 7, y_pos + 0.3)

# Frequency branch
draw_block(ax, 1, y_pos - 0.5, 3, 0.7, 'Frequency Branch\nConv2D (5×1, 256)\nDeeper frequency\npatterns',
          color_freq, fontsize=8)

# Temporal branch
draw_block(ax, 6, y_pos - 0.5, 3, 0.7, 'Temporal Branch\nConv2D (1×5, 256)\nDeeper temporal\npatterns',
          color_temp, fontsize=8)

# Merge arrows
draw_arrow(ax, 2.5, y_pos - 0.5, 4.5, y_pos - 1.3)
draw_arrow(ax, 7.5, y_pos - 0.5, 5.5, y_pos - 1.3)

# Fusion
y_pos = 8.6
draw_block(ax, 2.5, y_pos, 5, 0.6, 'Fusion Layer\nConcatenate + Conv2D (1×1, 256)',
          color_fusion, fontsize=9)

# Downsample + Residual
y_pos = 7.6
draw_block(ax, 2.5, y_pos, 5, 0.5, 'Downsample (MaxPool 2×2) + Projection Shortcut',
          color_fusion, fontsize=8)
ax.text(8.5, y_pos + 0.25, 'Output:\n32×27', fontsize=7, style='italic', color='gray')
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)

# Global Average Pooling
y_pos = 6.5
draw_block(ax, 2, y_pos, 6, 0.7, 'Global Average Pooling\n(32 × 27) → (256,)',
          color_pool, fontsize=10, fontweight='bold')
ax.text(8.5, y_pos + 0.35, 'Spatial\nreduction', fontsize=8, style='italic', color='gray')
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)

# Dropout
y_pos = 5.3
draw_block(ax, 2.5, y_pos, 5, 0.5, 'Dropout (p=0.5)',
          color_pool, fontsize=9)
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)

# Classifier
y_pos = 4.2
draw_block(ax, 2, y_pos, 6, 0.7, 'Fully Connected Layer\nLinear(256 → 18)',
          color_classifier, fontsize=10, fontweight='bold')
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)

# Softmax
y_pos = 3.0
draw_block(ax, 2, y_pos, 6, 0.6, 'Softmax Activation',
          color_classifier, fontsize=10)
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)

# Output
y_pos = 1.9
draw_block(ax, 2, y_pos, 6, 0.6, 'Output: Species Probabilities\n(18 classes)',
          color_input, fontsize=11, fontweight='bold')

# Legend
y_legend = 0.8
legend_items = [
    ('Input/Output', color_input),
    ('Stem Convolution', color_stem),
    ('Frequency Path', color_freq),
    ('Temporal Path', color_temp),
    ('Fusion/Downsample', color_fusion),
    ('Pooling/Regularization', color_pool),
    ('Classifier', color_classifier)
]

x_legend = 0.5
for i, (label, color) in enumerate(legend_items):
    if i == 4:  # Start second column
        x_legend = 5.5
        y_legend = 0.8

    ax.add_patch(mpatches.Rectangle((x_legend, y_legend - 0.15), 0.3, 0.2,
                                    facecolor=color, edgecolor='black'))
    ax.text(x_legend + 0.4, y_legend - 0.05, label, fontsize=8, va='center')
    y_legend -= 0.3

# Key features box
ax.text(5, 0.1, 'Key Features: Asymmetric Kernels | Dual-Path Processing | Residual Connections | Parameter Efficient',
       ha='center', fontsize=9, style='italic',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))

plt.tight_layout()
plt.savefig('/Volumes/Dev/Bird-Call-Detection/deepnet/architecture_diagram.png',
           dpi=300, bbox_inches='tight', facecolor='white')
print("Architecture diagram saved to: deepnet/architecture_diagram.png")
