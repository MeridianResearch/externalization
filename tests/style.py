from matplotlib.colors import hsv_to_rgb
from typing import List, Tuple

def orange_gradient(n_layers):

    colors = []
    for i in range(n_layers):
        position = i / (n_layers - 1) if n_layers > 1 else 0
        
        r = 0.8 + (0.95 - 0.8) * position
        g = 0.35 + (0.85 - 0.35) * position
        b = 0.1 + (0.7 - 0.1) * position
        
        colors.append((r, g, b))
    
    return colors


# Font sizes for charts
TITLE_FONTSIZE = 26
LABEL_FONTSIZE = 22
TICK_FONTSIZE = 20
ANNOTATION_FONTSIZE = 18