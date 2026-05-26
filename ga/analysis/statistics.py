from dataclasses import dataclass, asdict
import csv
import numpy as np
from pathlib import Path
import json

from ga.population import Population
from ga.individual import Individual
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

    def __str__(self):
        return (
            f"Generation: {self.generation}\n"
            f"Best:  {self.best_fitness:9.4f}\n"
            f"Mean:  {self.mean_fitness:9.4f}\n"
            f"Worst: {self.worst_fitness:9.4f}\n"
            f"Std:   {self.std_fitness:9.4f}\n"
            f"Large Hospitals: {self.n_hospitals_large}\n"
            f"Small Hospitals: {self.n_hospitals_small}"
        )


class GAStatistics:

    def __init__(self, record_individual_history:bool = False):
        self.history: list[GenerationStats] = []
        self.rih = record_individual_history
        if self.rih:
            self.worst_history: list[Individual] = []
            self.best_history: list[Individual] = []

    def record(
        self, 
        generation : int, 
        population : Population,
    ):

        fitness_values = np.array([
            ind.fitness for ind in population.individuals
        ])

        best = population.best()
        worst = population.worst()
        if self.rih:
            self.worst_history.append(worst.clone())
            self.best_history.append(best.clone())
        n_large = sum(1 for t in best.genome if t[0] == 2)
        n_small = len(best.genome) - n_large 

        stats = GenerationStats(
            generation=generation,
            best_fitness=np.min(fitness_values),
            mean_fitness=np.mean(fitness_values),
            worst_fitness=np.max(fitness_values),
            std_fitness=np.std(fitness_values),
            n_hospitals_large=n_large,
            n_hospitals_small=n_small
        )
        print(stats)
        self.history.append(stats)

    def save_csv(
        self, 
        run_dir: str | Path,
        do_plot: bool,
        record_individual_history: True,
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
        
        if record_individual_history:
            self.save_individual_history(run_dir=run_dir)
    
    def save_individual_history(self, run_dir):
        b_json_path = run_dir / "recordings_best.json"

        with open(b_json_path, "w") as f:
            json.dump([asdict(b.sim_records) for b in self.best_history], f, indent=4)

        w_json_path = run_dir / "recordings_worst.json"

        with open(w_json_path, "w") as f:
            json.dump([asdict(w.sim_records) for w in self.worst_history], f, indent=4)

