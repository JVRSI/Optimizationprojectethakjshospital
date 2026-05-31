import numpy as np

from ga import GAConfig, GeneticAlgorithm, BasicGenerator, GravityGenerator, ParallelEvaluator
from ga.selection import RouletteSelection, TruncateSelection
from ga.variation import EvolutionaryVariation, MicroGAVariation, ClassicVariation
from ga.analysis.statistics import GAStatistics
from gaFactory import GAFactory
from Simulation import SimConfig



def runs(size, record_history_of_best_and_worst, cities_matrix, seed=40):
    gc = GAConfig(
        n_generations=1000,
        initial_population_size=200,
        population_size=200,
        genome_size= size,
        mean_hospital_large=45,
        mean_hospital_small=70,
        collect_performance_data=True,
        plot_images=True,
        record_individual_history=record_history_of_best_and_worst,
        n_parents=40,
        n_hospital_types=2, #not everywhere is support for variable hospital types
        mutation_strategy="mutable_wandering", #{wandering, mutable_wandering, single_point, single_point_equal_opportunity}
        wandering_mutation_sigma=10,
        probability_of_mutation=0.08,
        crossover_strategy="grid", #{single_grid, grid, single_break}
        n_crossovers=10,
        probability_of_crossover=0.95,
        do_random_restarts=True,
        n_best_to_keep=10,
        n_steps_of_no_improvement_to_converge=20,
        max_time_to_run_s=1.5*60*60,
    )
    return [
        GAFactory(
            ga_config = gc,
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
            idea="check different hospital capacities",
            sim_config = SimConfig(
                SEED=seed,
                END_DAYS=20,
                CAPACITYL_N = 60,
                CAPACITYL_U = 0,
                CAPACITYS_N = 0,
                CAPACITYS_U = 10,
                COSTL=1500,
                COSTS=250,
                TOTALCOST=37000,
                SICK_RATE_U=2e-5,
                SICK_RATE_N=8e-5,
                PATIENT_DAYS_U=3,
                PATIENT_DAYS_N=7,
                URGENCY_N=1,
                URGENCY_U=2,
                BASE_SURVIVAL_PROB_U=0.995,
                BASE_SURVIVAL_PROB_N=0.999,
                DISTANCE_PENALTY_U=0.0015,
                DISTANCE_PENALTY_N=0.0002,
                SURVIVAL_NOISE_STD_U=0.01,
                SURVIVAL_NOISE_STD_N=0.003,
            )
        ),
        #-----------------------------------------------------------------------------
        GAFactory(
            ga_config = gc,
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
            idea="check different hospital capacities",
            sim_config = SimConfig(
                SEED=seed,
                END_DAYS=20,
                CAPACITYL_N = 60,
                CAPACITYL_U = 5,
                CAPACITYS_N = 0,
                CAPACITYS_U = 5,
                COSTL=1500,
                COSTS=250,
                TOTALCOST=37000,
                SICK_RATE_U=2e-5,
                SICK_RATE_N=8e-5,
                PATIENT_DAYS_U=3,
                PATIENT_DAYS_N=7,
                URGENCY_N=1,
                URGENCY_U=2,
                BASE_SURVIVAL_PROB_U=0.995,
                BASE_SURVIVAL_PROB_N=0.999,
                DISTANCE_PENALTY_U=0.0015,
                DISTANCE_PENALTY_N=0.0002,
                SURVIVAL_NOISE_STD_U=0.01,
                SURVIVAL_NOISE_STD_N=0.003,
            )
        ),
        #-----------------------------------------------------------------------------
        GAFactory(
            ga_config = gc,
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
            idea="check different hospital capacities",
            sim_config = SimConfig(
                SEED=seed,
                END_DAYS=20,
                CAPACITYL_N = 50,
                CAPACITYL_U = 0,
                CAPACITYS_N = 0,
                CAPACITYS_U = 10,
                COSTL=1500,
                COSTS=250,
                TOTALCOST=37000,
                SICK_RATE_U=2e-5,
                SICK_RATE_N=8e-5,
                PATIENT_DAYS_U=3,
                PATIENT_DAYS_N=7,
                URGENCY_N=1,
                URGENCY_U=2,
                BASE_SURVIVAL_PROB_U=0.995,
                BASE_SURVIVAL_PROB_N=0.999,
                DISTANCE_PENALTY_U=0.0015,
                DISTANCE_PENALTY_N=0.0002,
                SURVIVAL_NOISE_STD_U=0.01,
                SURVIVAL_NOISE_STD_N=0.003,
            )
        ),
        #-----------------------------------------------------------------------------
        GAFactory(
            ga_config = gc,
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
            idea="check different hospital capacities",
            sim_config = SimConfig(
                SEED=seed,
                END_DAYS=20,
                CAPACITYL_N = 50,
                CAPACITYL_U = 0,
                CAPACITYS_N = 10,
                CAPACITYS_U = 10,
                COSTL=1500,
                COSTS=250,
                TOTALCOST=37000,
                SICK_RATE_U=2e-5,
                SICK_RATE_N=8e-5,
                PATIENT_DAYS_U=3,
                PATIENT_DAYS_N=7,
                URGENCY_N=1,
                URGENCY_U=2,
                BASE_SURVIVAL_PROB_U=0.995,
                BASE_SURVIVAL_PROB_N=0.999,
                DISTANCE_PENALTY_U=0.0015,
                DISTANCE_PENALTY_N=0.0002,
                SURVIVAL_NOISE_STD_U=0.01,
                SURVIVAL_NOISE_STD_N=0.003,
            )
        ),
        #-----------------------------------------------------------------------------
        
    ]