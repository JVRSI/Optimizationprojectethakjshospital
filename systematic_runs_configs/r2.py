import numpy as np

from ga import GAConfig, GeneticAlgorithm, BasicGenerator, GravityGenerator, ParallelEvaluator
from ga.selection import RouletteSelection, TruncateSelection
from ga.variation import EvolutionaryVariation, MicroGAVariation, ClassicVariation
from ga.analysis.statistics import GAStatistics
from gaFactory import GAFactory

def runs(size, record_history_of_best_and_worst, cities_matrix):
    return [
        GAFactory(
            ga_config = GAConfig(
                n_generations=1000,
                initial_population_size=100,
                population_size=100,
                genome_size= size,
                mean_hospital_large=45,
                mean_hospital_small=70,
                collect_performance_data=True,
                plot_images=True,
                record_individual_history=record_history_of_best_and_worst,
                n_parents=20,
                n_hospital_types=2, #not everywhere is support for variable hospital types
                mutation_strategy="mutable_wandering", #{wandering, mutable_wandering, single_point, single_point_equal_opportunity}
                wandering_mutation_sigma=16,
                probability_of_mutation=0.08,
                crossover_strategy="grid", #{single_grid, grid, single_break}
                n_crossovers=10,
                probability_of_crossover=0.95,
                do_random_restarts=True,
                n_best_to_keep=10,
                n_steps_of_no_improvement_to_converge=20,
                max_time_to_run_s=4*60*60,
            ),
            generator_factory = lambda config, seed: GravityGenerator(
                rng=np.random.default_rng(seed),
                config=config,
                cities_matrix=cities_matrix
            ),
            selector_factory = lambda n_parents, seed: RouletteSelection(
                n_parents=n_parents,
                rng=np.random.default_rng(seed)
            ),
            variator_factory = lambda config, seed: ClassicVariation(
                ga_config=config,
                rng=np.random.default_rng(seed)
            ),
            generator_name="GravityGenerator",
            selector_name="RouletteSelection",
            variator_name="ClassicVariation",
            idea="Long running mutable wandering",
        ),
        #-----------------------------------------------------------------------------
        # 1 long single point with EO mutation
        GAFactory(
            ga_config = GAConfig(
                n_generations=1000,
                initial_population_size=100,
                population_size=100,
                genome_size= size,
                mean_hospital_large=45,
                mean_hospital_small=70,
                collect_performance_data=True,
                plot_images=True,
                record_individual_history=record_history_of_best_and_worst,
                n_parents=20,
                n_hospital_types=2, #not everywhere is support for variable hospital types
                mutation_strategy="single_point_equal_opportunity", #{wandering, mutable_wandering, single_point, single_point_equal_opportunity}
                wandering_mutation_sigma=16,
                probability_of_mutation=0.08,
                crossover_strategy="grid", #{single_grid, grid, single_break}
                n_crossovers=10,
                probability_of_crossover=0.95,
                do_random_restarts=True,
                n_best_to_keep=10,
                n_steps_of_no_improvement_to_converge=20,
                max_time_to_run_s=4*60*60,
            ),
            generator_factory = lambda config, seed: GravityGenerator(
                rng=np.random.default_rng(seed),
                config=config,
                cities_matrix=cities_matrix
            ),
            selector_factory = lambda n_parents, seed: RouletteSelection(
                n_parents=n_parents,
                rng=np.random.default_rng(seed)
            ),
            variator_factory = lambda config, seed: ClassicVariation(
                ga_config=config,
                rng=np.random.default_rng(seed)
            ),
            generator_name="GravityGenerator",
            selector_name="RouletteSelection",
            variator_name="ClassicVariation",
            idea="Long running SPMWEO",
        )
    ]