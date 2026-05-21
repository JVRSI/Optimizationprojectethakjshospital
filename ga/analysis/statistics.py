from dataclasses import dataclass, asdict
import csv
import numpy as np
from pathlib import Path

from ga.population import Population
from ga.analysis.plotting import plot_fitness


@dataclass
class GenerationStats:
    generation: int
    best_fitness : float
    mean_fitness : float
    worst_fitness : float
    std_fitness : float
    n_hospitals_large : int
    n_hospitals_small : int


class GAStatistics:

    def __init__(self):
        self.history: list[GenerationStats] = []

    def record(
        self, 
        generation : int, 
        population : Population
    ):

        fitness_values = np.array([
            ind.fitness for ind in population.individuals
        ])
        print(fitness_values)

        best = population.best()
        n_large = sum(1 for t in best.genome if t[0] == 1)
        n_small = len(best.genome) - n_large 

        stats = GenerationStats(
            generation=generation,
            best_fitness=np.max(fitness_values),
            mean_fitness=np.mean(fitness_values),
            worst_fitness=np.min(fitness_values),
            std_fitness=np.std(fitness_values),
            n_hospitals_large=n_large,
            n_hospitals_small=n_small
        )

        self.history.append(stats)

    def save_csv(
        self, 
        run_dir: str | Path,
        do_plot: bool
    ):

        run_dir.mkdir(parents=True, exist_ok=True)

        csv_path = run_dir / "recordings.csv"

        with open(csv_path, "w", newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames = list(asdict(self.history[0]).keys())
            )

            writer.writeheader()

            for stat in self.history:
                writer.writerow(asdict(stat))

        if do_plot:
            plot_fitness(csv_path,run_dir)