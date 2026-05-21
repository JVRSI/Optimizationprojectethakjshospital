import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
import json


from paths import DATA_DIR, RUNS_DIR

from ga import GAConfig, GeneticAlgorithm, BasicGenerator, ParallelEvaluator
from ga.selection import RouletteSelection
from ga.variation import EvolutionaryVariation
from ga.analysis.statistics import GAStatistics



from Simulation import Simulation, SimConfig
from Simulation.entities import City



with open(DATA_DIR / "gov_data" / "cities_list_reduced_from_root_rounded.pkl", "rb") as f:
    cities_list = pickle.load(f)


size = (219,345)
seed = 42
rng = np.random.default_rng(seed)



##############################################################################################
#GA config
ga_config = GAConfig(
    initial_population_size=100,
    population_size=200,
    genome_size= size,
    mean_hospital_large=30,
    mean_hospital_small=50,
    collect_performance_data=True,
    plot_images=True,
    n_parents=5,
    n_hospital_types=2,
    mutation_strategy="wandering", #{wandering, mutable_wandering, single_point, single_point_equal_opportunity}
    wandering_mutation_sigma=6,
    probability_of_mutation=0.01,
    crossover_strategy="single_grid", #{single_grid, grid, single_point}
    n_crossovers=10,
    probability_of_crossover=0.95
)




###############################################################################################
#Sim config
sim_config = SimConfig(
    SEED=seed,
    END_DAYS=20,
    CAPACITYL=100,
    CAPACITYS=2,
    COSTL=10,
    COSTS=5,
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



genome_generator = BasicGenerator(rng=rng,config=ga_config)
selector = RouletteSelection(n_parents=ga_config.n_parents,rng=rng)
variator = EvolutionaryVariation(rng=rng,ga_config=ga_config)
evaluator = ParallelEvaluator(sim_config=sim_config,cities=cities_list,n_workers=16)
ga_stats = GAStatistics()



genetic_algorithm = GeneticAlgorithm(
    ga_config,
    genome_generator=genome_generator,
    selection=selector,
    variation=variator,
    evaluator=evaluator,
    statistics=ga_stats,
    rng=rng
)

best = genetic_algorithm.run(50)

 # save run result, config and stats (in a new folder within runs)
dir_name = f"{selector.__class__.__name__}_{variator.__class__.__name__}_{genome_generator.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{evaluator.__class__.__name__}"
dir_path = Path(RUNS_DIR) / dir_name

# stats and plots
if ga_config.collect_performance_data:
    ga_stats.save_csv(dir_path, ga_config.plot_images)

# config
ga_config_dict = asdict(ga_config)
with open(dir_path / "ga_config.json", "w") as f:
    json.dump(ga_config_dict, f, indent=4)

sim_config_dict = asdict(sim_config)
with open(dir_path / "ga_config.json", "w") as f:
    json.dump(sim_config_dict, f, indent=4)

# result
best_dict = best.to_dict()
with open(dir_path / "best.json", "w") as f:
    json.dump(best_dict, f, indent=4)