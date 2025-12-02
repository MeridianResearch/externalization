from matplotlib.colors import hsv_to_rgb
from typing import List, Tuple

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-1 range)"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def orange_gradient(n_layers):
    palette = [
        '#b63b02',
        '#e95d0d',
        '#F3701B',
        '#FDA761',
        '#FDD8B3',
        '#FFF5EB'
    ]
    
    colors = []
    for i in range(n_layers):
        hex_color = palette[i % len(palette)]
        colors.append(hex_to_rgb(hex_color))
    
    return colors

# Font sizes for charts
TITLE_FONTSIZE = 26
LABEL_FONTSIZE = 22
TICK_FONTSIZE = 20
ANNOTATION_FONTSIZE = 14