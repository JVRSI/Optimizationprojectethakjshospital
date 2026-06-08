import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Union
from datetime import datetime
import argparse

from paths import RUNS_DIR, RUNS_LOCAL, DATA_DIR
try:
    from colors import PALETTE_DARKISH_VAR, PALETTE_GREENISH
except ImportError:
    from evaluating.colors import PALETTE_DARKISH_VAR, PALETTE_GREENISH
try:
    from helpers import plot_genome, get_latest_run
except ImportError:
    from evaluating.helpers import plot_genome, get_latest_run

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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--newest", action="store_true")
    args = parser.parse_args()

    #run_dir = RUNS_LOCAL / "RouletteSelection_EvolutionaryVariation_GravityGenerator_2026-05-22_14-09-44_ParallelEvaluator"
    #run_dir = RUNS_DIR / "0_systematic_r4" / "8_1_RouletteSelection_ClassicVariation_GravityGenerator_2026-06-01_12-48-38_ParallelEvaluator"
    #run_dir = RUNS_DIR  / ""

    run_dir = RUNS_DIR / "0_systematic_r6" 

    run_dir = next(run_dir.glob("0_0_*"))

    if args.newest:
        run_dir = get_latest_run(RUNS_LOCAL)



    json_file = run_dir / "best.json"
    json_file = run_dir / "recordings_best.json"
    json_file = run_dir / "last_generation.json"

    with open(json_file, 'r') as f:
        best = json.load(f)["individuals"][20]

    genome = best["genome"]
    fitness = best["fitness"]
    plot_genome(genome=genome, show_plot=True, fitness=fitness)



