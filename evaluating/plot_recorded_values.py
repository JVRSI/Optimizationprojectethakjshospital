import json
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from paths import RUNS_DIR, RUNS_LOCAL

try:
    from helpers import plot_genome, create_gif
except ImportError:
    from evaluating.helpers import plot_genome, create_gif



if __name__ == "__main__":

    #run_dir = RUNS_LOCAL / "RouletteSelection_EvolutionaryVariation_GravityGenerator_2026-05-22_14-09-44_ParallelEvaluator"
    #run_dir = RUNS_DIR / "0_systematic_r3" / "0_1_RouletteSelection_ClassicVariation_GravityGenerator_2026-05-31_05-43-03_ParallelEvaluator"
    #run_dir = RUNS_DIR  / ""

    run_dir = RUNS_DIR / "0_systematic_r4" 

    run_dir = next(run_dir.glob("28_4_*"))

    print(run_dir)

    p = "recordings_best"
    json_file = run_dir / f"{p}.json"

    with open(json_file, 'r') as f:
        best_history = json.load(f)
    
    sim_records = []
    for best in best_history:
        t = best["sim_records"]
        t["fitness"] = best["fitness"]
        sim_records.append(t)
    
    df = pd.DataFrame(sim_records)
    df_norm = (df - df.min()) / (df.max() - df.min())

    df_fitness_calc_norm = df.copy()
    df_fitness_calc_norm["death_rate"] *= 0.6
    df_fitness_calc_norm["not_admitted_rate"] *= 0.8
    df_fitness_calc_norm["urgent_death_rate"] *= 0.5
    df_fitness_calc_norm["urgent_not_admitted_rate"] *= 0.5
    df_fitness_calc_norm["normalized_admitted_distance"] *= 0
    df_fitness_calc_norm["normalized_not_survived_distance"] *= 0
    df_fitness_calc_norm["normalized_admitted_choice_rank"] *= 0
    df_fitness_calc_norm["normalized_not_survived_choice_rank"] *= 0
    df_fitness_calc_norm["normalized_unused_hospitals"] *= 0
    df_fitness_calc_norm["normalized_cost"] *= 0.02



    df_fitness_calc_norm["reconstructed_fitness"] = df_fitness_calc_norm[[
        "death_rate",
        "not_admitted_rate",
        "urgent_death_rate",
        "urgent_not_admitted_rate",
        "normalized_admitted_distance",
        "normalized_not_survived_distance",
        "normalized_admitted_choice_rank",
        "normalized_not_survived_choice_rank",
        "normalized_unused_hospitals",
        "normalized_cost",
    ]].sum(axis=1)
    df_fitness_calc_norm["small_summed"] = df_fitness_calc_norm[[
        "urgent_death_rate",
        "urgent_not_admitted_rate",
        "normalized_admitted_distance",
        "normalized_not_survived_distance",
        "normalized_admitted_choice_rank",
        "normalized_not_survived_choice_rank",
    ]].sum(axis=1)

    df_fitness_calc_norm[[
        "fitness",
        "death_rate",
        "not_admitted_rate",
        "urgent_death_rate",
        "urgent_not_admitted_rate",
        "normalized_admitted_distance",
        "normalized_not_survived_distance",
        "normalized_admitted_choice_rank",
        "normalized_not_survived_choice_rank",
        "normalized_unused_hospitals",
        #"small_summed",
        "normalized_cost",
        #"reconstructed_fitness"
    ]].plot(kind='line', legend=True, figsize=(12,6))
    
    if False:
        df[[
            "not_admitted_count",
            "not_survived_count",
            "admitted_count",
            "not_survived_urgent",
            "not_survived_nonurgent",
            "not_admitted_urgent",
            "not_admitted_nonurgent",
            "admitted_urgent",
            "admitted_nonurgent",
            #"total_travel_distance",
        ]].plot(kind='line', legend=True, figsize=(12,6))

    



    plt.ylabel("Value")
    plt.title("Simulation Records")
    plt.tight_layout()
    plt.show()




