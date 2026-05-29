import numpy as np

from ga import GAConfig, GeneticAlgorithm, BasicGenerator, GravityGenerator, ParallelEvaluator
from ga.selection import RouletteSelection, TruncateSelection
from ga.variation import EvolutionaryVariation, MicroGAVariation, ClassicVariation
from ga.analysis.statistics import GAStatistics
from gaFactory import GAFactory

def runs(size, record_history_of_best_and_worst, cities_matrix):
    return [
        #-------------------------------------------------
        #- So far best one
        # GAFactory(
        #     ga_config = GAConfig(
        #         n_generations=100,
        #         initial_population_size=100,
        #         population_size=100,
        #         genome_size= size,
        #         mean_hospital_large=30,
        #         mean_hospital_small=50,
        #         collect_performance_data=True,
        #         plot_images=True,
        #         record_individual_history=record_history_of_best_and_worst,
        #         n_parents=20,
        #         n_hospital_types=2, #not everywhere is support for variable hospital types
        #         mutation_strategy="mutable_wandering", #{wandering, mutable_wandering, single_point, single_point_equal_opportunity}
        #         wandering_mutation_sigma=16,
        #         probability_of_mutation=0.08,
        #         crossover_strategy="grid", #{single_grid, grid, single_point}
        #         n_crossovers=10,
        #         probability_of_crossover=0.95,
        #         do_random_restarts=False,
        #         n_best_to_keep=10,
        #         n_steps_of_no_improvement_to_converge=5,
        #         max_time_to_run_s=7200,
        #     ),
        #     generator_factory = lambda config, seed: GravityGenerator(
        #         rng=np.random.default_rng(seed),
        #         config=config,
        #         cities_matrix=cities_matrix
        #     ),
        #     selector_factory = lambda n_parents, seed: RouletteSelection(
        #         n_parents=n_parents,
        #         rng=np.random.default_rng(seed)
        #     ),
        #     variator_factory = lambda config, seed: ClassicVariation(
        #         ga_config=config,
        #         rng=np.random.default_rng(seed)
        #     ),
        #     seed=4727638,
        #     generator_name="GravityGenerator",
        #     selector_name="RouletteSelection",
        #     variator_name="ClassicVariation",
        #     idea="Worked best so far",
        # ),
        #-----------------------------------------------------------------------------
        # 0
        GAFactory(
            ga_config = GAConfig(
                n_generations=100,
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
                crossover_strategy="grid", #{single_grid, grid, single_point}
                n_crossovers=10,
                probability_of_crossover=0.95,
                do_random_restarts=True,
                n_best_to_keep=10,
                n_steps_of_no_improvement_to_converge=10,
                max_time_to_run_s=5400,
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
            idea="Worked best so far, with random restart, but new sim config",
        ),
        #-----------------------------------------------------------------------------
        # 1 no gravity initialization
        GAFactory(
            ga_config = GAConfig(
                n_generations=100,
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
                crossover_strategy="grid", #{single_grid, grid, single_point}
                n_crossovers=10,
                probability_of_crossover=0.95,
                do_random_restarts=True,
                n_best_to_keep=10,
                n_steps_of_no_improvement_to_converge=10,
                max_time_to_run_s=5400,
            ),
            generator_factory = lambda config, seed: BasicGenerator(
                rng=np.random.default_rng(seed),
                config=config
            ),
            selector_factory = lambda n_parents, seed: RouletteSelection(
                n_parents=n_parents,
                rng=np.random.default_rng(seed)
            ),
            variator_factory = lambda config, seed: ClassicVariation(
                ga_config=config,
                rng=np.random.default_rng(seed)
            ),
            generator_name="BasicGenerator",
            selector_name="RouletteSelection",
            variator_name="ClassicVariation",
            idea="Same as 0, but without gravity initialization",
        ),
        #-----------------------------------------------------------------------------
        # 2 Truncate selection
        GAFactory(
            ga_config = GAConfig(
                n_generations=100,
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
                crossover_strategy="grid", #{single_grid, grid, single_point}
                n_crossovers=10,
                probability_of_crossover=0.95,
                do_random_restarts=True,
                n_best_to_keep=10,
                n_steps_of_no_improvement_to_converge=10,
                max_time_to_run_s=5400,
            ),
            generator_factory = lambda config, seed: GravityGenerator(
                rng=np.random.default_rng(seed),
                config=config,
                cities_matrix=cities_matrix
            ),
            selector_factory = lambda n_parents, seed: TruncateSelection(
                n_parents=n_parents,
                rng=np.random.default_rng(seed)
            ),
            variator_factory = lambda config, seed: ClassicVariation(
                ga_config=config,
                rng=np.random.default_rng(seed)
            ),
            generator_name="GravityGenerator",
            selector_name="TruncateSelection",
            variator_name="ClassicVariation",
            idea="0 with truncate selection",
        ),
        #-----------------------------------------------------------------------------
        # 3 single point mutation
        GAFactory(
            ga_config = GAConfig(
                n_generations=100,
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
                mutation_strategy="single_point", #{wandering, mutable_wandering, single_point, single_point_equal_opportunity}
                wandering_mutation_sigma=16,
                probability_of_mutation=0.08,
                crossover_strategy="grid", #{single_grid, grid, single_point}
                n_crossovers=10,
                probability_of_crossover=0.95,
                do_random_restarts=True,
                n_best_to_keep=10,
                n_steps_of_no_improvement_to_converge=10,
                max_time_to_run_s=5400,
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
            idea="0 with single point mutation",
        ),
        #-----------------------------------------------------------------------------
        # 4 single point with equal opportunity
        GAFactory(
            ga_config = GAConfig(
                n_generations=100,
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
                crossover_strategy="grid", #{single_grid, grid, single_point}
                n_crossovers=10,
                probability_of_crossover=0.95,
                do_random_restarts=True,
                n_best_to_keep=10,
                n_steps_of_no_improvement_to_converge=10,
                max_time_to_run_s=5400,
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
            idea="0 with single point mutation with equal opportunity",
        ),
        #-----------------------------------------------------------------------------
        # 5 Single point crossover
        GAFactory(
            ga_config = GAConfig(
                n_generations=100,
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
                crossover_strategy="single_point", #{single_grid, grid, single_point}
                n_crossovers=10,
                probability_of_crossover=0.95,
                do_random_restarts=True,
                n_best_to_keep=10,
                n_steps_of_no_improvement_to_converge=10,
                max_time_to_run_s=5400,
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
            idea="Worked best so far, with random restart, but new sim config",
        ),
        #-----------------------------------------------------------------------------
        # 6 low mutation
        GAFactory(
            ga_config = GAConfig(
                n_generations=100,
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
                probability_of_mutation=0.008,
                crossover_strategy="grid", #{single_grid, grid, single_point}
                n_crossovers=10,
                probability_of_crossover=0.95,
                do_random_restarts=True,
                n_best_to_keep=10,
                n_steps_of_no_improvement_to_converge=10,
                max_time_to_run_s=2700,
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
            idea="0 with low mutation",
        ),
        #-----------------------------------------------------------------------------
        # 7 high mutation
        GAFactory(
            ga_config = GAConfig(
                n_generations=100,
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
                probability_of_mutation=0.15,
                crossover_strategy="grid", #{single_grid, grid, single_point}
                n_crossovers=10,
                probability_of_crossover=0.95,
                do_random_restarts=True,
                n_best_to_keep=10,
                n_steps_of_no_improvement_to_converge=10,
                max_time_to_run_s=2700,
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
            idea="0 with high mutation",
        ),
    ]