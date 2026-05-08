import pandas as pd
import matplotlib.pyplot as plt


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