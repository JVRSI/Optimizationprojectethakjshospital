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
        interpolation="none"
    )

    x = 129
    y = 136
    r = 50

    # Kreis hinzufügen
    circle = Circle(
        (x, y),      # Mittelpunkt
        r,           # Radius
        fill=False,  # nur Rand
        edgecolor="red",
        linewidth=2
    )
    circle2 = Circle(
        (x, y),      # Mittelpunkt
        25,           # Radius
        fill=False,  # nur Rand
        edgecolor="red",
        linewidth=2
    )

    circle3 = Circle(
        (x, y),      # Mittelpunkt
        2,           # Radius
        fill=True,  # nur Rand
        linewidth=2,
    )

    plt.gca().add_patch(circle)
    plt.gca().add_patch(circle2)
    plt.gca().add_patch(circle3)

    plt.show()



