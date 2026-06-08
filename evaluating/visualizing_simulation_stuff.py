import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Union
from datetime import datetime
import argparse
from matplotlib.patches import Circle

from paths import RUNS_DIR, RUNS_LOCAL, DATA_DIR
try:
    from colors import PALETTE_DARKISH_VAR, PALETTE_GREENISH
except ImportError:
    from evaluating.colors import PALETTE_DARKISH_VAR, PALETTE_GREENISH
try:
    from helpers import plot_genome, get_latest_run, load_matrix, basicCol
except ImportError:
    from evaluating.helpers import plot_genome, get_latest_run, load_matrix, basicCol





if __name__ == "__main__":

    matrix = load_matrix()

    matrix = matrix**0.5


    plt.figure(figsize=(13, 8), facecolor=basicCol.background)

    plt.imshow(
        matrix,
        origin="lower",
        cmap=basicCol.custom_cmap,
        interpolation="none",
        #aspect="equal"
    )

    x = 128
    y = 134
    r = 50

    # Kreis hinzufügen
    circle = Circle(
        (x, y),      # Mittelpunkt
        r,           # Radius
        fill=False,  # nur Rand
        edgecolor="#1b475d",
        linewidth=2
    )
    circle2 = Circle(
        (x, y),      # Mittelpunkt
        25,           # Radius
        fill=False,  # nur Rand
        edgecolor="#b4bd62",
        linewidth=2
    )

    circle3 = Circle(
        (x, y),      # Mittelpunkt
        2,           # Radius
        linewidth=2,
        facecolor="#fad564"
    )

    plt.gca().add_patch(circle)
    plt.gca().add_patch(circle2)
    plt.gca().add_patch(circle3)


    plt.xticks(range(0, matrix.shape[1], 25))
    plt.yticks(range(0, matrix.shape[0], 25))
    plt.gca().set_axisbelow(True)   
    plt.grid(True, color="gray", linewidth=0.5, alpha=0.3)
    
    plt.show()



