from dataclasses import dataclass
from typing import Tuple

@dataclass
class GAConfig:
    n_generations : int = 10

    initial_population_size: int = 1000
    population_size: int = 200

    genome_size: Tuple[float, float] = (10, 20) # (height, width)
    mean_hospital_large: int = 30
    mean_hospital_small: float = 50

    collect_performance_data: bool = True # to safe to csv and plot
    plot_images : bool = True # to also create nice plots (only plots if data is collected)
    record_individual_history : bool = False

    n_parents : int = 5

    n_hospital_types : int = 2  # shouldn't this be simulation config, or some global config

    #mutation
    mutation_strategy : str = "wandering"  #only if Variation strategy allows selecting mutation Strategy {wandering, mutable_wandering, single_point, single_point_equal_opportunity}
    wandering_mutation_sigma : float = 6
    probability_of_mutation : float = 1e-2  #only if Variation where mutation is tied to a probability

    #crossover
    crossover_strategy : str = "single_grid" #only if variation strategy allows selection {single_grid, grid, single_point}
    n_crossovers : int = 10
    probability_of_crossover : float = 0.95  #only if Variation where mutation is tied to a probability

    #convergence
    do_random_restarts : bool = True #if not reached max iterations, restart if has converged
    n_best_to_keep : int = 5 #number of best individuals to keep when doing random restart
    n_steps_of_no_improvement_to_converge : int = 5

    max_time_to_run_s : float = 7200
