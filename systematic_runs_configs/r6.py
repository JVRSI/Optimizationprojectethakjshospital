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
        mean_hospital_large=30,
        mean_hospital_small=120,
        random_amount = False,
        collect_performance_data=True,
        plot_images=True,
        record_individual_history=record_history_of_best_and_worst,
        n_parents=10,
        n_elites=2,
        n_hospital_types=2, #not everywhere is support for variable hospital types
        mutation_strategy="teleport", #{wandering, mutable_wandering, single_point, single_point_equal_opportunity, teleport}
        wandering_mutation_sigma=10,
        probability_of_mutation=0.3,
        crossover_strategy="positive_grid", #{single_grid, grid, single_break, positive_gene_exchange, positive_grid}
        n_crossovers=10,
        probability_of_crossover=0.95,
        p_exchange=0.3,
        do_random_restarts=True,
        n_best_to_keep=10,
        n_steps_of_no_improvement_to_converge=5,
        max_time_to_run_s=2*60*60,
        slop_threshold=0.005,
        std_threshold=0.002
    )
    sc = SimConfig(
        SEED=seed,
        END_DAYS=30,
        CAPACITYL_N = 85,
        CAPACITYL_U = 0,
        CAPACITYS_N = 0,
        CAPACITYS_U = 6,
        COSTL=1500,
        COSTS=1000,
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
        not_admitted_rate_factor=0.6,
        urgent_death_rate_factor=0.3,
        urgent_not_admitted_rate_factor=0.3,
        normalized_admitted_distance_factor=0,
        normalized_not_survived_distance_factor=0,
        normalized_admitted_choice_rank_factor=0,
        normalized_not_survived_choice_rank_factor=0,
        normalized_unused_hospitals_factor=0,
        normalized_cost_factor=0.02,
        bad_coverage_factor=0,
        )
    return [
        #-----------------------------------------------------------------------------------------------------------
        # 0 MC Random, fluid
        GAFactory(
            ga_config = replace(gc,
                random_amount=True,
                max_time_to_run_s = 0.5*60*60,
                n_best_to_keep=1,
                n_steps_of_no_improvement_to_converge=0,
            ),
            generator_factory = lambda config, seed: BasicGenerator(rng=np.random.default_rng(seed),config=config,),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: ClassicVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="BasicGenerator",
            selector_name="TruncateSelection",
            variator_name="ClassicVariation",
            idea="MC Random, fluid",
            sim_config=sc,
        ),
        #-----------------------------------------------------------------------------------------------------------
        # 1 MC Gravity, fluid
        GAFactory(
            ga_config = replace(gc,
                random_amount=True,
                max_time_to_run_s = 0.5*60*60,
                n_best_to_keep=1,
                n_steps_of_no_improvement_to_converge=0,
            ),
            generator_factory = lambda config, seed: GravityGenerator(rng=np.random.default_rng(seed),config=config,cities_matrix=cities_matrix),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: ClassicVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="GravityGenerator",
            selector_name="TruncateSelection",
            variator_name="ClassicVariation",
            idea="MC Gravity, fluid",
            sim_config=sc,
        ),
        #-----------------------------------------------------------------------------------------------------------
        # 2 Fluid MGA
        GAFactory(
            ga_config = replace(gc,
                random_amount=True,
                n_elites=0,
                n_parents=98,
                crossover_strategy="grid",
            ),
            generator_factory = lambda config, seed: GravityGenerator(rng=np.random.default_rng(seed),config=config,cities_matrix=cities_matrix),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: MicroGAVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="GravityGenerator",
            selector_name="TruncateSelection",
            variator_name="MicroGAVariation",
            idea="",
            sim_config=sc,
        ),
        #-----------------------------------------------------------------------------------------------------------
        # 3 Fluid Classic, m:0.1, c:0.95
        GAFactory(
            ga_config = replace(gc,
                random_amount=True,
                crossover_strategy="grid",
                mutation_strategy="single_point_equal_opportunity",
                probability_of_mutation=0.1,
                probability_of_crossover=0.95
            ),
            generator_factory = lambda config, seed: GravityGenerator(rng=np.random.default_rng(seed),config=config,cities_matrix=cities_matrix),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: ClassicVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="GravityGenerator",
            selector_name="TruncateSelection",
            variator_name="ClassicVariation",
            idea="",
            sim_config=sc,
        ),
        #-----------------------------------------------------------------------------------------------------------
        # 4 Fluid Classic, m:0.6554, c:0.8694
        GAFactory(
            ga_config = replace(gc,
                random_amount=True,
                crossover_strategy="grid",
                mutation_strategy="single_point_equal_opportunity",
                probability_of_mutation=0.6554,
                probability_of_crossover=0.8694,
            ),
            generator_factory = lambda config, seed: GravityGenerator(rng=np.random.default_rng(seed),config=config,cities_matrix=cities_matrix),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: ClassicVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="GravityGenerator",
            selector_name="TruncateSelection",
            variator_name="ClassicVariation",
            idea="",
            sim_config=sc,
        ),
        #-----------------------------------------------------------------------------------------------------------
        # 5 Fluid Classic, m:0.8694, c:0.6554
        GAFactory(
            ga_config = replace(gc,
                random_amount=True,
                crossover_strategy="grid",
                mutation_strategy="single_point_equal_opportunity",
                probability_of_mutation=0.8694,
                probability_of_crossover=0.6554
            ),
            generator_factory = lambda config, seed: GravityGenerator(rng=np.random.default_rng(seed),config=config,cities_matrix=cities_matrix),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: ClassicVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="GravityGenerator",
            selector_name="TruncateSelection",
            variator_name="ClassicVariation",
            idea="",
            sim_config=sc,
        ),
        #-----------------------------------------------------------------------------------------------------------
        # 6 Fluid Classic, m:0.95, c:0.1
        GAFactory(
            ga_config = replace(gc,
                random_amount=True,
                crossover_strategy="grid",
                mutation_strategy="single_point_equal_opportunity",
                probability_of_mutation=0.95,
                probability_of_crossover=0.1
            ),
            generator_factory = lambda config, seed: GravityGenerator(rng=np.random.default_rng(seed),config=config,cities_matrix=cities_matrix),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: ClassicVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="GravityGenerator",
            selector_name="TruncateSelection",
            variator_name="ClassicVariation",
            idea="",
            sim_config=sc,
        ),
        #-----------------------------------------------------------------------------------------------------------
        # 7 Fluid ES
        GAFactory(
            ga_config = replace(gc,
                random_amount=True,
                mutation_strategy="single_point_equal_opportunity",
            ),
            generator_factory = lambda config, seed: GravityGenerator(rng=np.random.default_rng(seed),config=config,cities_matrix=cities_matrix),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: EvolutionaryVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="GravityGenerator",
            selector_name="TruncateSelection",
            variator_name="EvolutionaryVariation",
            idea="",
            sim_config=sc,
        ),
        ####################################################################################################################
        #Fixed stuff
        ####################################################################################################################
        #-----------------------------------------------------------------------------------------------------------
        # 8 Fixed MC Random
        GAFactory(
            ga_config = replace(gc,
                max_time_to_run_s = 0.5*60*60,
                n_best_to_keep=1,
                n_steps_of_no_improvement_to_converge=0,
            ),
            generator_factory = lambda config, seed: BasicGenerator(rng=np.random.default_rng(seed),config=config,),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: ClassicVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="BasicGenerator",
            selector_name="TruncateSelection",
            variator_name="ClassicVariation",
            idea="Fixed MC Random",
            sim_config=sc,
        ),
        #-----------------------------------------------------------------------------------------------------------
        # 9 Fixed MC Gravity
        GAFactory(
            ga_config = replace(gc,
                max_time_to_run_s = 0.5*60*60,
                n_best_to_keep=1,
                n_steps_of_no_improvement_to_converge=0,
            ),
            generator_factory = lambda config, seed: GravityGenerator(rng=np.random.default_rng(seed),config=config,cities_matrix=cities_matrix),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: ClassicVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="GravityGenerator",
            selector_name="TruncateSelection",
            variator_name="ClassicVariation",
            idea="MC Gravity, fluid",
            sim_config=sc,
        ),
        #-----------------------------------------------------------------------------------------------------------
        # 10 Fixed MGA
        GAFactory(
            ga_config = replace(gc,
                n_elites=0,
                n_parents=98,
            ),
            generator_factory = lambda config, seed: GravityGenerator(rng=np.random.default_rng(seed),config=config,cities_matrix=cities_matrix),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: MicroGAVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="GravityGenerator",
            selector_name="TruncateSelection",
            variator_name="MicroGAVariation",
            idea="",
            sim_config=sc,
        ),
        #-----------------------------------------------------------------------------------------------------------
        # 11 Fixed Classic, m:0.1, c:0.95
        GAFactory(
            ga_config = replace(gc,
                probability_of_mutation=0.1,
                probability_of_crossover=0.95
            ),
            generator_factory = lambda config, seed: GravityGenerator(rng=np.random.default_rng(seed),config=config,cities_matrix=cities_matrix),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: ClassicVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="GravityGenerator",
            selector_name="TruncateSelection",
            variator_name="ClassicVariation",
            idea="",
            sim_config=sc,
        ),
        #-----------------------------------------------------------------------------------------------------------
        # 12 Fixed Classic, m:0.6554, c:0.8694
        GAFactory(
            ga_config = replace(gc,
                probability_of_mutation=0.6554,
                probability_of_crossover=0.8694,
            ),
            generator_factory = lambda config, seed: GravityGenerator(rng=np.random.default_rng(seed),config=config,cities_matrix=cities_matrix),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: ClassicVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="GravityGenerator",
            selector_name="TruncateSelection",
            variator_name="ClassicVariation",
            idea="",
            sim_config=sc,
        ),
        #-----------------------------------------------------------------------------------------------------------
        # 13 Fixed Classic, m:0.8694, c:0.6554
        GAFactory(
            ga_config = replace(gc,
                probability_of_mutation=0.8694,
                probability_of_crossover=0.6554
            ),
            generator_factory = lambda config, seed: GravityGenerator(rng=np.random.default_rng(seed),config=config,cities_matrix=cities_matrix),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: ClassicVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="GravityGenerator",
            selector_name="TruncateSelection",
            variator_name="ClassicVariation",
            idea="",
            sim_config=sc,
        ),
        #-----------------------------------------------------------------------------------------------------------
        # 14 Fixed Classic, m:0.95, c:0.1
        GAFactory(
            ga_config = replace(gc,
                probability_of_mutation=0.95,
                probability_of_crossover=0.1
            ),
            generator_factory = lambda config, seed: GravityGenerator(rng=np.random.default_rng(seed),config=config,cities_matrix=cities_matrix),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: ClassicVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="GravityGenerator",
            selector_name="TruncateSelection",
            variator_name="ClassicVariation",
            idea="",
            sim_config=sc,
        ),
        #-----------------------------------------------------------------------------------------------------------
        # 15 Fixed ES
        GAFactory(
            ga_config = gc,
            generator_factory = lambda config, seed: GravityGenerator(rng=np.random.default_rng(seed),config=config,cities_matrix=cities_matrix),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: EvolutionaryVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="GravityGenerator",
            selector_name="TruncateSelection",
            variator_name="EvolutionaryVariation",
            idea="",
            sim_config=sc,
        ),
        #################################################################################################
        # #hospital initialization
        #################################################################################################
        #--------------------------------------------------------------------------------
        # 16 few initializations
        GAFactory(
            ga_config = replace(gc,
                random_amount=True,
                crossover_strategy="grid",
                mutation_strategy="single_point_equal_opportunity",
                probability_of_mutation=0.6554,
                probability_of_crossover=0.8694,
                mean_hospital_large=10,
                mean_hospital_small=50,
            ),
            generator_factory = lambda config, seed: GravityGenerator(rng=np.random.default_rng(seed),config=config,cities_matrix=cities_matrix),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: ClassicVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="GravityGenerator",
            selector_name="TruncateSelection",
            variator_name="ClassicVariation",
            idea="",
            sim_config=sc,
        ),
        #---------------------------------------------------------------------------------
        # 17 lots initializations
        GAFactory(
            ga_config = replace(gc,
                random_amount=True,
                crossover_strategy="grid",
                mutation_strategy="single_point_equal_opportunity",
                probability_of_mutation=0.6554,
                probability_of_crossover=0.8694,
                mean_hospital_large=50,
                mean_hospital_small=180,
            ),
            generator_factory = lambda config, seed: GravityGenerator(rng=np.random.default_rng(seed),config=config,cities_matrix=cities_matrix),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: ClassicVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="GravityGenerator",
            selector_name="TruncateSelection",
            variator_name="ClassicVariation",
            idea="",
            sim_config=sc,
        ),
    ]