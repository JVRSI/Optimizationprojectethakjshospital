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



def fluid_vs_liquid():

    version = "r6"
    season = "2"

    run_dir = RUNS_DIR / f"0_systematic_{version}"

    p = "recordings_best.json"
    

    runs_fluid = [
        next(run_dir.glob(f"{season}_0_*") ,None),
        next(run_dir.glob(f"{season}_1_*") ,None),
        next(run_dir.glob(f"{season}_2_*") ,None),
        next(run_dir.glob(f"{season}_3_*") ,None),
        next(run_dir.glob(f"{season}_4_*") ,None),
        next(run_dir.glob(f"{season}_5_*") ,None),
        next(run_dir.glob(f"{season}_6_*") ,None),
        next(run_dir.glob(f"{season}_7_*") ,None),
    ]
    runs_fixed = [
        next(run_dir.glob(f"{season}_8_*") ,None),
        next(run_dir.glob(f"{season}_9_*") ,None),
        next(run_dir.glob(f"{season}_10_*"),None),
        next(run_dir.glob(f"{season}_11_*"),None),
        next(run_dir.glob(f"{season}_12_*"),None),
        next(run_dir.glob(f"{season}_13_*"),None),
        next(run_dir.glob(f"{season}_14_*"),None),
        next(run_dir.glob(f"{season}_15_*"),None),
    ]

    runs = runs_fluid


    best_dict = []
    for r in runs:
        if r is None:
            best_dict.append(None)
            continue
        with open(run_dir / r / p, "r") as f:
            best_dict.append(json.load(f))
    
    gc_dict = []
    for r, rf in zip(runs,runs_fixed):
        t = r
        if t is None:
            t = rf
        if t is None:
            gc_dict.append(None)
            continue
        with open(run_dir / t / "ga_config.json", "r") as f:
            gc_dict.append(json.load(f))

    best_dict_f = []
    for r in runs_fixed:
        if r is None:
            best_dict_f.append(None)
            continue
        with open(run_dir / r / p, "r") as f:
            best_dict_f.append(json.load(f))
    
    legends = [
        "Random MC",
        "Gravity MC",
        "Micro GA",
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


    #fluid
    if True:

        #Random MC
        if best_dict[0] is not None:
            fitness = [d["fitness"] for d in best_dict[0]]
            ax.plot(
                fitness,
                label=legends[0],
                color="black",
                linestyle="--",
            )

        # Gravity MC
        if best_dict[1] is not None:
            fitness = [d["fitness"] for d in best_dict[1]]
            ax.plot(
                fitness,
                label=legends[1],
                color="black",
                linestyle=":",
            )

        for idx, color in zip(range(2, 8), classic_colors):
            if best_dict[idx] is None:
                continue
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

    if True:

        #Random MC
        if best_dict_f[0] is not None:
            fitness = [d["fitness"] for d in best_dict_f[0]]
            ax.plot(
                fitness,
                label=legends[0],
                color="black",
                linestyle="--",
            )

        # Gravity MC
        if best_dict_f[1] is not None:
            fitness = [d["fitness"] for d in best_dict_f[1]]
            ax.plot(
                fitness,
                label=legends[1],
                color="black",
                linestyle=":",
            )
        

        for idx, color in zip(range(2, 8), classic_colors2):
            if best_dict_f[idx] is None:
                continue
            fitness = [d["fitness"] for d in best_dict_f[idx]]
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


def mutation_check():

    version = "r7"
    season = "3"

    run_dir = RUNS_DIR / f"0_systematic_{version}"

    p = "recordings_best.json"
    

    

    fig, ax = plt.subplots(figsize=(10, 6))
    classic_colors = plt.cm.viridis(np.linspace(0, 1, 6))
    classic_colors2 = plt.cm.Spectral(np.linspace(0, 1, 6))
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


    runs_fluid = [
        next(run_dir.glob(f"{season}_2_*") ,None),
        next(run_dir.glob(f"{season}_3_*") ,None),
        next(run_dir.glob(f"{season}_4_*") ,None),
        next(run_dir.glob(f"{season}_5_*") ,None),
    ]

    best_dict_fluid = []
    for r in runs_fluid:
        if r is None:
            best_dict_fluid.append(None)
            continue
        with open(run_dir / r / p, "r") as f:
            best_dict_fluid.append(json.load(f))
    
    gc_dict_fluid = []
    for r in runs_fluid:
        with open(run_dir / r / "ga_config.json", "r") as f:
            gc_dict_fluid.append(json.load(f))

    legends_fluid = [
        f"fluid m:{gc_dict_fluid[0]['mutation_strategy']}",
        f"fluid m:{gc_dict_fluid[1]['mutation_strategy']}",
        f"fluid m:{gc_dict_fluid[2]['mutation_strategy']}",
        f"fluid m:{gc_dict_fluid[3]['mutation_strategy']}",
    ]


    #fluid
    if True:
        for idx, color in zip(range(0, 4), classic_colors):
            if best_dict_fluid[idx] is None:
                continue
            fitness = [d["fitness"] for d in best_dict_fluid[idx]]
            ax.plot(
                fitness,
                label=legends_fluid[idx],
                color=color,
                linewidth=1.5,
                marker=markers[idx],
                markevery=(5, 50+5*idx),
                #ms = 8,
            )

    
    runs_fixed = [
        next(run_dir.glob(f"{season}_6_*") ,None),
        next(run_dir.glob(f"{season}_7_*") ,None),
        next(run_dir.glob(f"{season}_8_*") ,None),
    ]

    best_dict_fixed = []
    for r in runs_fixed:
        if r is None:
            best_dict_fixed.append(None)
            continue
        with open(run_dir / r / p, "r") as f:
            best_dict_fixed.append(json.load(f))
    
    gc_dict_fixed = []
    for r in runs_fixed:
        with open(run_dir / r / "ga_config.json", "r") as f:
            gc_dict_fixed.append(json.load(f))

    legends_fixed = [
        f"fixed m:{gc_dict_fixed[0]['mutation_strategy']}",
        f"fixed m:{gc_dict_fixed[1]['mutation_strategy']}",
        f"fixed m:{gc_dict_fixed[2]['mutation_strategy']}",
    ]

    #fixed
    if True:
        for idx, color in zip(range(0, 3), classic_colors2):
            if best_dict_fixed[idx] is None:
                continue
            fitness = [d["fitness"] for d in best_dict_fixed[idx]]
            ax.plot(
                fitness,
                label=legends_fixed[idx],
                color=color,
                linewidth=1.5,
                marker=markers[idx+4],
                markevery=(5, 50+5*idx),
                #ms = 8,
            )


    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def crossover_check():

    version = "r8"
    #season = "3"

    run_dir = RUNS_DIR / f"0_systematic_{version}"

    p = "recordings_best.json"
    

    

    fig, ax = plt.subplots(figsize=(10, 6))
    classic_colors = plt.cm.viridis(np.linspace(0, 1, 6))
    classic_colors2 = plt.cm.Spectral(np.linspace(0, 1, 6))
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


    runs_fluid = [
        next(run_dir.glob(f"1_0_*") ,None),
        next(run_dir.glob(f"1_1_*") ,None),
        next(run_dir.glob(f"0_2_*") ,None),
        next(run_dir.glob(f"0_3_*") ,None),
    ]

    best_dict_fluid = []
    for r in runs_fluid:
        if r is None:
            best_dict_fluid.append(None)
            continue
        with open(run_dir / r / p, "r") as f:
            best_dict_fluid.append(json.load(f))
    
    gc_dict_fluid = []
    for r in runs_fluid:
        with open(run_dir / r / "ga_config.json", "r") as f:
            gc_dict_fluid.append(json.load(f))

    legends_fluid = [
        f"fluid c:{gc_dict_fluid[0]['crossover_strategy']}",
        f"fluid c:{gc_dict_fluid[1]['crossover_strategy']}",
        f"fluid c:{gc_dict_fluid[2]['crossover_strategy']} {gc_dict_fluid[2]['n_crossovers']}",
        f"fluid c:{gc_dict_fluid[3]['crossover_strategy']} {gc_dict_fluid[3]['n_crossovers']}",
    ]


    #fluid
    if True:
        for idx, color in zip(range(0, 4), classic_colors):
            if best_dict_fluid[idx] is None:
                continue
            fitness = [d["fitness"] for d in best_dict_fluid[idx]]
            ax.plot(
                fitness,
                label=legends_fluid[idx],
                color=color,
                linewidth=1.5,
                marker=markers[idx],
                markevery=(5, 50+5*idx),
                #ms = 8,
            )

    
    runs_fixed = [
        next(run_dir.glob(f"0_4_*") ,None),
        next(run_dir.glob(f"0_5_*") ,None),
        next(run_dir.glob(f"1_6_*") ,None),
    ]

    best_dict_fixed = []
    for r in runs_fixed:
        if r is None:
            best_dict_fixed.append(None)
            continue
        with open(run_dir / r / p, "r") as f:
            best_dict_fixed.append(json.load(f))
    
    gc_dict_fixed = []
    for r in runs_fixed:
        with open(run_dir / r / "ga_config.json", "r") as f:
            gc_dict_fixed.append(json.load(f))

    legends_fixed = [
        f"fixed c:{gc_dict_fixed[0]['crossover_strategy']}",
        f"fixed c:{gc_dict_fixed[1]['crossover_strategy']} {gc_dict_fixed[1]['n_crossovers']}",
        f"fixed c:{gc_dict_fixed[2]['crossover_strategy']} {gc_dict_fixed[2]['n_crossovers']}",
    ]

    #fixed
    if True:
        for idx, color in zip(range(0, 3), classic_colors2):
            if best_dict_fixed[idx] is None:
                continue
            fitness = [d["fitness"] for d in best_dict_fixed[idx]]
            ax.plot(
                fitness,
                label=legends_fixed[idx],
                color=color,
                linewidth=1.5,
                marker=markers[idx+4],
                markevery=(5, 50+5*idx),
                #ms = 8,
            )


    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    fluid_vs_liquid()
    mutation_check()
    crossover_check()


"""
Hypothesis why fixed is slower than fluid even if the #hospitals stays mostly constant: In fixed, a teleported hospital is like ranomly create one and randomly delet one, in mutation with EO, one can generate a new hospital somewhere and "delete" hospitals in crossover with the chance to choose from multiple ways of deleting a hospital. This is supported by the fact that in ES with fixed hospitals performs relatively good, whereas the fluid ES (no crossover) is worse than with crossover (in same range as fixed). though tis is a plot with one random state, maybe other random state other results but toy runs with other random states give similar results.
"""




