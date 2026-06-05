import pandas as pd
import matplotlib.pyplot as plt
import json

from Simulation import SimConfig


def plot_fitness(csv_path, output_dir):

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))

    plt.plot(df["generation"], df["best_fitness"], label="Best")
    plt.plot(df["generation"], df["mean_fitness"], label="Mean")
    plt.plot(df["generation"], df["worst_fitness"], label="Worst")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    plt.title("Fitness over Generations")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(output_dir / "fitness_plot.png")

    plt.close()

def plot_fitness_components(json_path, output_dir, sim_config:SimConfig):
    with open(json_path, 'r') as f:
        best_history = json.load(f)
    
    sim_records = []
    for best in best_history:
        t = best["sim_records"]
        t["fitness"] = best["fitness"]
        sim_records.append(t)
    
    df = pd.DataFrame(sim_records)

    df_fitness_calc_norm = df.copy()
    df_fitness_calc_norm["death_rate"] *= sim_config.death_rate_factor
    df_fitness_calc_norm["not_admitted_rate"] *= sim_config.not_admitted_rate_factor
    df_fitness_calc_norm["urgent_death_rate"] *= sim_config.urgent_death_rate_factor
    df_fitness_calc_norm["urgent_not_admitted_rate"] *= sim_config.urgent_not_admitted_rate_factor
    df_fitness_calc_norm["normalized_admitted_distance"] *= sim_config.normalized_admitted_distance_factor
    df_fitness_calc_norm["normalized_not_survived_distance"] *= sim_config.normalized_not_survived_distance_factor
    df_fitness_calc_norm["normalized_admitted_choice_rank"] *= sim_config.normalized_admitted_choice_rank_factor
    df_fitness_calc_norm["normalized_not_survived_choice_rank"] *= sim_config.normalized_not_survived_choice_rank_factor
    df_fitness_calc_norm["normalized_unused_hospitals"] *= sim_config.normalized_unused_hospitals_factor
    df_fitness_calc_norm["normalized_cost"] *= sim_config.normalized_cost_factor
    df_fitness_calc_norm["bad_coverage"] *= sim_config.bad_coverage_factor


    df_fitness_calc_norm[[
        "fitness",
        "death_rate",
        "not_admitted_rate",
        "urgent_death_rate",
        "urgent_not_admitted_rate",
        #"normalized_admitted_distance",
        #"normalized_not_survived_distance",
        #"normalized_admitted_choice_rank",
        #"normalized_not_survived_choice_rank",
        #"normalized_unused_hospitals",
        "normalized_cost",
        #"bad_coverage",
    ]].plot(kind='line', legend=True, figsize=(12,6))


    plt.ylabel("Value")
    plt.title("Simulation Records")
    plt.tight_layout()
    plt.savefig(output_dir / "fitness_components_plot.png")
    plt.close()