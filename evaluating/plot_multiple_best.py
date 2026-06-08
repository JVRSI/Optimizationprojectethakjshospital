import json
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from paths import RUNS_DIR, RUNS_LOCAL

try:
    from helpers import plot_genome, create_gif
except ImportError:
    from evaluating.helpers import plot_genome, create_gif


if __name__ == "__main__":

    version = "r6"
    season = "1"

    run_dir = RUNS_DIR / f"0_systematic_{version}"

    p = "recordings_best.json"
    

    runs_fluid = [
        next(run_dir.glob(f"{season}_0_*")),
        next(run_dir.glob(f"{season}_1_*")),
        next(run_dir.glob(f"{season}_2_*")),
        next(run_dir.glob(f"{season}_3_*")),
        next(run_dir.glob(f"{season}_4_*")),
        next(run_dir.glob(f"{season}_5_*")),
        next(run_dir.glob(f"{season}_6_*")),
        next(run_dir.glob(f"{season}_7_*")),
    ]
    runs_fixed = [
        #next(run_dir.glob(f"{season}_8_*")),
        next(run_dir.glob(f"{season}_9_*")),
        next(run_dir.glob(f"{season}_10_*")),
        next(run_dir.glob(f"{season}_11_*")),
        next(run_dir.glob(f"{season}_12_*")),
        next(run_dir.glob(f"{season}_13_*")),
        next(run_dir.glob(f"{season}_14_*")),
        next(run_dir.glob(f"{season}_15_*")),
    ]

    runs = runs_fluid


    best_dict = []
    for r in runs:
        with open(run_dir / r / p, "r") as f:
            best_dict.append(json.load(f))
    
    gc_dict = []
    for r in runs:
        with open(run_dir / r / "ga_config.json", "r") as f:
            gc_dict.append(json.load(f))

    best_dict_f = []
    for r in runs_fixed:
        with open(run_dir / r / p, "r") as f:
            best_dict_f.append(json.load(f))
    

    a = f"{gc_dict[3]['probability_of_mutation']:.4f}".rstrip("0").rstrip(".")

    legends = [
        "Random MC",
        "Gravity MC",
        "Micro GA",
        #f"Classic m:{gc_dict[3]['probability_of_mutation']:1.4g} c:{gc_dict[3]['probability_of_crossover']:1.4g}",
        f"Classic m:{gc_dict[3]['probability_of_mutation']:1.4g} c:{gc_dict[3]['probability_of_crossover']:1.4g}",
        f"Classic m:{gc_dict[4]['probability_of_mutation']:1.4g} c:{gc_dict[4]['probability_of_crossover']:1.4g}",
        f"Classic m:{gc_dict[5]['probability_of_mutation']:1.4g} c:{gc_dict[5]['probability_of_crossover']:1.4g}",
        f"Classic m:{gc_dict[6]['probability_of_mutation']:1.4g} c:{gc_dict[6]['probability_of_crossover']:1.4g}",
        "ES",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    classic_colors = plt.cm.viridis(np.linspace(0, 1, 6))
    classic_colors2 = plt.cm.plasma(np.linspace(0, 1, 6))
    markers = [
        "o",
        "x",
        "+",
        "*",
        "s",
        "^",
        "v",
        "D",
        ".",
    ]


    #Random MC
    # fitness = [d["fitness"] for d in best_dict[0]]
    # ax.plot(
    #     fitness,
    #     label=legends[0],
    #     color="black",
    #     linestyle="--",
    # )

    # Gravity MC
    fitness = [d["fitness"] for d in best_dict[1]]
    ax.plot(
        fitness,
        label=legends[1],
        color="black",
        linestyle=":",
    )

    for idx, color in zip(range(2, 8), classic_colors):
        fitness = [d["fitness"] for d in best_dict[idx]]
        ax.plot(
            fitness,
            label=legends[idx],
            color=color,
            linewidth=1.5,
            marker=markers[idx],
            markevery=(5, 50+5*idx),
            #ms = 8,
        )

    for idx, color in zip(range(2, 8), classic_colors2):
        fitness = [d["fitness"] for d in best_dict_f[idx-1]]
        ax.plot(
            fitness,
            label=legends[idx],
            color=color,
            linewidth=1.5,
            marker=markers[idx],
            markevery=(5, 50+5*idx),
            #ms = 8,
        )

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()





