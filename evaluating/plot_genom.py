import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Union

from paths import RUNS_DIR, RUNS_LOCAL, DATA_DIR
from colors import PALETTE_DARKISH_VAR, PALETTE_GREENISH


#color stuff

'''
palette = PALETTE_DARKISH_VAR
custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", [palette["a"],palette["c"]])
c_hospital_l = palette["d"]
c_hospital_s = palette["b"]
c_grid       = None
c_face       = palette["a"]
c_text       = palette["f"]
'''

palette = PALETTE_GREENISH
custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", ["white",palette["d"]])
c_hospital_l = palette["c"]
c_hospital_s = palette["b"]
c_grid       = None
c_face       = "white"
c_text       = "black"



save_plot = False

run_dir = RUNS_LOCAL / "RouletteSelection_MicroGAVariation_BasicGenerator_2026-05-22_00-08-44_ParallelEvaluator"
#run_dir = RUNS_DIR  / ""

json_file = run_dir / "best.json"

with open(json_file, 'r') as f:
    best = json.load(f)

genome = best["genome"]
fitness = best["fitness"]


matrix_file = DATA_DIR / "gov_data" / "Daten Matrix Reduced.csv"


def load_matrix():
    # Load reduced matrix robustly
    df = pd.read_csv(matrix_file, header=None, sep=None, engine="python")

    # Convert all cells to numeric values
    df = df.apply(pd.to_numeric, errors="coerce")

    # Replace non-numeric / empty cells with 0
    df = df.fillna(0)

    return df.to_numpy(dtype=float)

def plot_genome(
        genome: Union[list[list], list[tuple[int, int, int]]],
        path: Path = None,
        show_plot: bool = True,
        fitness: float = None,
    ):
    """
    Given a genome, plots the hospitals on the population heat map

    Can handle list of lists with 3 element, or list of tuples with 3 elements

    If path is specified, plot will be saved there.
    """

    sizes, row, col = map(np.array, zip(*genome))
    l = 1


    matrix = load_matrix()

    matrix = matrix**0.5

    print(matrix.shape)

    dot_sizes = np.where(sizes == l, 60, 45)

    plt.figure(figsize=(13, 8), facecolor=c_face)

    plt.imshow(
        matrix,
        origin="lower",
        cmap=custom_cmap,
        interpolation="none"
    )

    major = sizes == l

    plt.scatter(
        col[~major],
        row[~major],
        s=dot_sizes[~major],
        c=c_hospital_s,
        edgecolors=None,
        linewidths=0.8,
        label="Hospitals"
    )


    plt.scatter(
        col[major],
        row[major],
        s=dot_sizes[major],
        facecolors=c_hospital_l,
        edgecolors=None,
        linewidths=1.5,
        label="Large hospitals"
    )

    if fitness is None:
        title = "Genome"
    else:
        title = f"Genome with fitness {fitness:4.4f}"
    #plt.xlabel("Reduced matrix x-coordinate")
    #plt.ylabel("Reduced matrix y-coordinate")

    #plt.xlim(0, matrix.shape[1])
    #plt.ylim(0, matrix.shape[0])


    #colors and formatting
    ax = plt.gca()
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if c_grid is not None:
        plt.grid(color=c_grid, linewidth=0.3, alpha=0.4)

    print(plt.rcParams["font.monospace"][0])

    plt.rcParams["text.color"] = c_text
    
    ax.text(
        0.0, 0.98,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
    )


    plt.legend(frameon=False,fontsize=11)
    plt.tight_layout()


    if path is not None:
        plt.savefig(path)

    if show_plot:
        plt.show()



if __name__ == "__main__":
    plot_genome(genome=genome, show_plot=True, fitness=fitness)



