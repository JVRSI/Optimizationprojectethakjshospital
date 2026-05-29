import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Union
from datetime import datetime
import imageio.v2 as imageio

from paths import RUNS_DIR, RUNS_LOCAL, DATA_DIR

try:
    from colors import PALETTE_DARKISH_VAR, PALETTE_GREENISH
except ImportError:
    from evaluating.colors import PALETTE_DARKISH_VAR, PALETTE_GREENISH


class GenomePlotColors:

    def __init__(
            self,
            background,
            custom_cmap,
            hospital_l,
            hospital_s,
            grid,
            text,
        ):
        self.background = background
        self.custom_cmap = custom_cmap
        self.hospital_l = hospital_l
        self.hospital_s = hospital_s
        self.grid = grid
        self.text = text
        pass

palette = PALETTE_GREENISH
basicCol = GenomePlotColors(
    background="white",
    custom_cmap=LinearSegmentedColormap.from_list("custom_gradient", ["white",palette["d"]]),
    hospital_l=palette["c"],
    hospital_s=palette["b"],
    grid=None,
    text="black"
)



def load_matrix():
    # Load reduced matrix robustly
    matrix_file = DATA_DIR / "gov_data" / "Daten Matrix Reduced.csv"
    df = pd.read_csv(matrix_file, header=None, sep=None, engine="python")

    # Convert all cells to numeric values
    df = df.apply(pd.to_numeric, errors="coerce")

    # Replace non-numeric / empty cells with 0
    df = df.fillna(0)

    return df.to_numpy(dtype=float)

def plot_genome(
        genome: Union[list[list], list[tuple[int, int, int]]],
        colors: GenomePlotColors = basicCol,
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
    l = 2


    matrix = load_matrix()

    matrix = matrix**0.5

    plt.figure(figsize=(13, 8), facecolor=colors.background)

    plt.imshow(
        matrix,
        origin="lower",
        cmap=colors.custom_cmap,
        interpolation="none"
    )

    major = sizes == l

    plt.scatter(
        col[major],
        row[major],
        s=90,
        facecolors=colors.hospital_l,
        edgecolors=None,
        linewidths=1.5,
        label="Large hospitals"
    )
    

    plt.scatter(
        col[~major],
        row[~major],
        s=45,
        c=colors.hospital_s,
        edgecolors=None,
        linewidths=0.8,
        label="Hospitals"
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

    if colors.grid is not None:
        plt.grid(color=colors.grid, linewidth=0.3, alpha=0.4)

    plt.rcParams["text.color"] = colors.text
    
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
        path.parent.mkdir(exist_ok=True)
        plt.savefig(path)

    if show_plot:
        plt.show()
    plt.close()

#######################################################################################################
# create gif

def create_gif(
        folder_path : Path,
        file_name_start : str,
        file_name_end : str,
        max_images : int = 1000
    ):
    """
    The loaded images will be
    f"{file_name_start}{i}{file_name_end}"

    in the specified folder
    """
    images = []

    for i in range(max_images):
        path = folder_path / f"{file_name_start}{i}{file_name_end}"

        if path.exists():
            images.append(imageio.imread(path))

    imageio.mimsave(
        folder_path / "animation.gif",
        images,
        duration=0.25,
    )

    

########################################################################################################
#load data from folder stuff

def get_run_datetime(path: Path) -> datetime:
    parts = path.name.split("_")

    date_part = parts[3]
    time_part = parts[4]

    return datetime.strptime(
        f"{date_part}_{time_part}",
        "%Y-%m-%d_%H-%M-%S"
    )


def get_latest_run(run_dir: Path) -> Path:

    runs = [
        p for p in run_dir.iterdir()
        if p.is_dir()
    ]

    return max(runs, key=get_run_datetime)