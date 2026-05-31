import json
import argparse
from tqdm import tqdm

from paths import RUNS_DIR, RUNS_LOCAL

try:
    from helpers import plot_genome, create_gif
except ImportError:
    from evaluating.helpers import plot_genome, create_gif



if __name__ == "__main__":

    #run_dir = RUNS_LOCAL / "RouletteSelection_EvolutionaryVariation_GravityGenerator_2026-05-22_14-09-44_ParallelEvaluator"
    run_dir = RUNS_DIR / "0_systematic_r2" / "3_0_RouletteSelection_ClassicVariation_GravityGenerator_2026-05-30_20-32-58_ParallelEvaluator"
    #run_dir = RUNS_DIR  / ""

    p = "recordings_best"
    json_file = run_dir / f"{p}.json"

    with open(json_file, 'r') as f:
        best_history = json.load(f)

    print("Plotting...")
    for i, best in tqdm(enumerate(best_history), total=len(best_history)):
        genome = best["genome"]
        fitness = best["fitness"]
        plot_genome(genome=genome, show_plot=False, fitness=fitness, path=run_dir/"local_plots"/f"genome_{p}_{i}.png")
    
    create_gif(run_dir/"local_plots", file_name_start=f"genome_{p}_",file_name_end=".png")



