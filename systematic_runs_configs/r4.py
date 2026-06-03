import numpy as np

from ga import GAConfig, GeneticAlgorithm, BasicGenerator, GravityGenerator, ParallelEvaluator
from ga.selection import RouletteSelection, TruncateSelection
from ga.variation import EvolutionaryVariation, MicroGAVariation, ClassicVariation
from ga.analysis.statistics import GAStatistics
from gaFactory import GAFactory
from Simulation import SimConfig
from dataclasses import replace



def runs(size, record_history_of_best_and_worst, cities_matrix, seed=40):
    gc = GAConfig(
        n_generations=1000,
        initial_population_size=100,
        population_size=100,
        genome_size= size,
        mean_hospital_large=35,
        mean_hospital_small=400,
        collect_performance_data=True,
        plot_images=True,
        record_individual_history=record_history_of_best_and_worst,
        n_parents=10,
        n_elites=10,
        n_hospital_types=2, #not everywhere is support for variable hospital types
        mutation_strategy="single_point_equal_opportunity", #{wandering, mutable_wandering, single_point, single_point_equal_opportunity, teleport}
        wandering_mutation_sigma=10,
        probability_of_mutation=1,
        crossover_strategy="grid", #{single_grid, grid, single_break, positive_gene_exchange}
        n_crossovers=10,
        probability_of_crossover=0,
        p_exchange=0.2,
        do_random_restarts=True,
        n_best_to_keep=10,
        n_steps_of_no_improvement_to_converge=5,
        max_time_to_run_s=0.7*60*60,
    )
    sc = SimConfig(
        SEED=seed,
        END_DAYS=40,
        CAPACITYL_N = 85,
        CAPACITYL_U = 0,
        CAPACITYS_N = 0,
        CAPACITYS_U = 6,
        COSTL=1500,
        COSTS=200,
        TOTALCOST=37000,
        SICK_RATE_U=0.000022,
        SICK_RATE_N=0.00004,
        PATIENT_DAYS_U=3,
        PATIENT_DAYS_N=7,
        URGENCY_N=1,
        URGENCY_U=2,
        BASE_SURVIVAL_PROB_U=0.995,
        BASE_SURVIVAL_PROB_N=0.999,
        DISTANCE_PENALTY_U=0.0015,
        DISTANCE_PENALTY_N=0.000004,
        SURVIVAL_NOISE_STD_U=0.01,
        SURVIVAL_NOISE_STD_N=0.003,
        death_rate_factor=0.6,
        not_admitted_rate_factor=0.8,
        urgent_death_rate_factor=0.3,
        urgent_not_admitted_rate_factor=0.3,
        normalized_admitted_distance_factor=0,
        normalized_not_survived_distance_factor=0,
        normalized_admitted_choice_rank_factor=0,
        normalized_not_survived_choice_rank_factor=0,
        normalized_unused_hospitals_factor=0,
        normalized_cost_factor=0.02,
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
            idea="Based on theoretical ideas from observations so far",
            sim_config=sc
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
            idea="pervious one with smaller capacities for small, but less cost",
            sim_config=replace(sc, COSTS=250)
        ),
        #-----------------------------------------------------------------------------
        GAFactory(
            ga_config = replace(gc, mutation_strategy="mutable_wandering"),
            generator_factory = lambda config, seed: GravityGenerator(
                rng=np.random.default_rng(seed),
                config=config,
                cities_matrix=cities_matrix
            ),
            selector_factory = lambda n_parents, seed: RouletteSelection(
                n_parents=n_parents,
                rng=np.random.default_rng(seed)
            ),
            variator_factory = lambda config, seed: EvolutionaryVariation(
                ga_config=config,
                rng=np.random.default_rng(seed)
            ),
            generator_name="GravityGenerator",
            selector_name="RouletteSelection",
            variator_name="ClassicVariation",
            idea="pervious one with smaller capacities and mutable wandering",
            sim_config=replace(sc,COSTS=400)
        ),
        #-----------------------------------------------------------------------------
        GAFactory(
            ga_config = replace(gc, mutation_strategy="wandering", crossover_strategy="positive_gene_exchange", mean_hospital_small = 150),
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
            variator_name="EvolutionaryVariation",
            idea="pervious one with smaller capacities for small, but less cost and mutable wandering",
            sim_config=replace(sc, COSTS=733 )
        ),
        #-----------------------------------------------------------------------------
        GAFactory(
            ga_config = replace(gc, mutation_strategy="wandering", crossover_strategy="positive_gene_exchange", mean_hospital_small=110),
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
            variator_name="EvolutionaryVariation",
            idea="pervious one with smaller capacities for small, but less cost and mutable wandering",
            sim_config=replace(sc, COSTS=1000 )
        ),
        #-----------------------------------------------------------------------------
        GAFactory(
            ga_config = replace(gc, mutation_strategy="teleport", crossover_strategy="positive_gene_exchange", mean_hospital_small=150),
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
            variator_name="EvolutionaryVariation",
            idea="pervious one with smaller capacities for small, but less cost and mutable wandering",
            sim_config=replace(sc, COSTS=733)
        ),
        #-----------------------------------------------------------------------------
        GAFactory(
            ga_config = replace(gc, mutation_strategy="teleport", crossover_strategy="positive_gene_exchange", mean_hospital_small=110),
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
            variator_name="EvolutionaryVariation",
            idea="pervious one with smaller capacities for small, but less cost and mutable wandering",
            sim_config=replace(sc, COSTS=1000 )
        ),
        #-----------------------------------------------------------------------------
    ]