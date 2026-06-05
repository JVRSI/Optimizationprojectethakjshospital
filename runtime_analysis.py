import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import json
import sys
import argparse
import cProfile
import pstats
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import math


from paths import DATA_DIR, RUNS_DIR, MATRIX_PATH, RUNS_LOCAL

from ga import GAConfig, GeneticAlgorithm, BasicGenerator, GravityGenerator, ParallelEvaluator
from ga.selection import RouletteSelection
from ga.variation import EvolutionaryVariation, MicroGAVariation, ClassicVariation
from ga.analysis.statistics import GAStatistics
from ga.individual import Individual


from testSim import matrix_to_city_dataframe, load_population_matrix



from Simulation import Simulation, SimConfig
from Simulation.entities import City



with open(DATA_DIR / "gov_data" / "cities_list_reduced_from_root_rounded_with_coverage.pkl", "rb") as f:
    cities_list = pickle.load(f)


size = (219,345)
seed = 42
rng = np.random.default_rng(seed)

n_workers = 15
record_history_of_best_and_worst = True

##############################################################################################
#GA config
ga_config = GAConfig(
    n_generations=2,
    initial_population_size=10,
    population_size=10,
    genome_size= size,
    mean_hospital_large=30,
    mean_hospital_small=50,
    collect_performance_data=True,
    plot_images=True,
    record_individual_history=record_history_of_best_and_worst,
    n_parents=4,
    n_hospital_types=2, #not everywhere is support for variable hospital types
    mutation_strategy="mutable_wandering", #{wandering, mutable_wandering, single_point, single_point_equal_opportunity}
    wandering_mutation_sigma=16,
    probability_of_mutation=0.08,
    crossover_strategy="grid", #{single_grid, grid, single_break}
    n_crossovers=10,
    probability_of_crossover=0.95,
    do_random_restarts=False,
    n_best_to_keep=10,
    n_steps_of_no_improvement_to_converge=5,
    max_time_to_run_s=7200,
)




###############################################################################################
#Sim config
sim_config = SimConfig(
    SEED=seed,
    END_DAYS=10,
    CAPACITYL_N=400,
    CAPACITYL_U=0,
    CAPACITYS_N=0,
    CAPACITYS_U=4,
    COSTL=1500,
    COSTS=250,
    TOTALCOST=37000,
    SICK_RATE_U=0.000022,
    SICK_RATE_N=0.0004,
    PATIENT_DAYS_U=3,
    PATIENT_DAYS_N=7,
    URGENCY_N=1,
    URGENCY_U=2,
    BASE_SURVIVAL_PROB_U=0.995,
    BASE_SURVIVAL_PROB_N=0.999,
    DISTANCE_PENALTY_U=0.0015,
    DISTANCE_PENALTY_N=0.000002,
    SURVIVAL_NOISE_STD_U=0.01,
    SURVIVAL_NOISE_STD_N=0.003,
)

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--n-generations", type=int, default=ga_config.n_generations)
parser.add_argument("-l", "--log-local", action="store_true")
parser.add_argument("-d", "--dont-store-stats", action="store_true")
args = parser.parse_args()

ga_config.n_generations = args.n_generations



cities_matrix = load_population_matrix(MATRIX_PATH)

genome_generator = GravityGenerator(rng=rng,config=ga_config,cities_matrix=cities_matrix)
selector = RouletteSelection(n_parents=ga_config.n_parents,rng=rng)
variator = ClassicVariation(rng=rng,ga_config=ga_config)
evaluator = ParallelEvaluator(sim_config=sim_config,cities=cities_list,n_workers=n_workers,cities_matrix=None, rng=rng)
ga_stats = GAStatistics()

store_run = False
runs_dir = RUNS_DIR

if args.dont_store_stats:
    store_run = False
if args.log_local:
    runs_dir = RUNS_LOCAL


genetic_algorithm = GeneticAlgorithm(
    ga_config,
    genome_generator=genome_generator,
    selection=selector,
    variation=variator,
    evaluator=evaluator,
    statistics=ga_stats,
    rng=rng
)

def main():

    # all your current main.py code goes here

    best, status = genetic_algorithm.run()

    print("-------------------------------------------------------------------------------")
    print(status)


    if store_run:

        # save run result, config and stats (in a new folder within runs)
        dir_name = f"{selector.__class__.__name__}_{variator.__class__.__name__}_{genome_generator.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{evaluator.__class__.__name__}"
        dir_path = runs_dir / dir_name

        # stats and plots
        if ga_config.collect_performance_data:
            ga_stats.save_csv(dir_path, ga_config.plot_images)

        # config
        ga_config_dict = asdict(ga_config)
        with open(dir_path / "ga_config.json", "w") as f:
            json.dump(ga_config_dict, f, indent=4)

        sim_config_dict = asdict(sim_config)
        with open(dir_path / "sim_config.json", "w") as f:
            json.dump(sim_config_dict, f, indent=4)

        # result
        best_dict = best.to_dict()
        with open(dir_path / "best.json", "w") as f:
            json.dump(best_dict, f, indent=4)

cities_matrix = load_population_matrix(MATRIX_PATH)

def sim_only():
    rng = np.random.default_rng(3)
    generator = GravityGenerator(rng=rng,config=ga_config,cities_matrix=cities_matrix)
    genome=generator()
    individual = Individual(genome=genome)
    sim = Simulation(start_pos=individual.genome,sc=sim_config,cities_list=cities_list,rng=rng,do_analysis=True)
    individual.fitness = sim.run(log=True)

    t = sim.duration

    return sim.get_result(), t, sim.survival_probability_per_distance, sim.survival_result_by_probability, sim.patient_by_place_and_survival

def sim_with_load(p:Path):
    rng = np.random.default_rng(42)
    with open(p / "recordings_best.json", 'r') as f:
        best = json.load(f)[-1]
    genome=[(g[0],g[1],g[2]) for g in best["genome"]]
    individual = Individual(genome=genome)
    with open(p / "sim_config.json", 'r') as f:
        sc_d = json.load(f)
    sc = SimConfig(**sc_d)
    sc.END_DAYS = 500
    sim = Simulation(start_pos=individual.genome,sc=sc,cities_list=cities_list,rng=rng,do_analysis=True)
    individual.fitness = sim.run(log=True)

    t = sim.duration

    return sim.get_result(), t, sim.survival_probability_per_distance, sim.survival_result_by_probability, sim.patient_by_place_and_survival


if __name__ == "__main__":

    #sim_only()

    #main()

    res, _, p_by_dist, res_of_p, p_by_place_survival = sim_with_load(runs_dir / "0_systematic_r4" / "28_4_TruncateSelection_ClassicVariation_GravityGenerator_2026-06-03_03-37-52_ParallelEvaluator")

    print(res.to_scalar())

    if True: #plot places where people survive or die
        p_u = [(p[1],p[2]) for p in p_by_place_survival if p[0] == sim_config.URGENCY_U]
        p_n = [(p[1],p[2]) for p in p_by_place_survival if p[0] == sim_config.URGENCY_N]
        

        
        def give_ratio(l):
            counts = defaultdict(lambda: [0, 0])

            for value, coord in l:
                counts[coord][1] += 1
                if value:
                    counts[coord][0] += 1

            return [
                (true_count / total_count, coord)
                for coord, (true_count, total_count) in counts.items()
            ], [
                total_count for _, (_, total_count) in counts.items()
            ]
        r_u, t_u = give_ratio(p_u)
        r_n, t_n = give_ratio(p_n)

        mt_u = max([t for t in t_u])
        mt_n = max([t for t in t_n])

        t_u = [(t/mt_u)**0.5 for t in t_u]
        t_n = [(t/mt_n)**0.5 for t in t_n]

        r, t = r_u, t_u

        plt.figure(figsize=(13, 8))

        # col = [coord[0] for _, coord in r]
        # row = [coord[1] for _, coord in r]
        # c = [ratio for ratio, _ in r]

        # scatter = plt.scatter(
        #     row,
        #     col,
        #     c=c,
        #     cmap="RdYlGn",  # Rot -> Gelb -> Grün
        #     vmin=0,
        #     vmax=1,
        #     s=20
        # )

        

        m = np.full(ga_config.genome_size, np.nan)
        a = np.zeros(ga_config.genome_size)

        for ((c, coord), alpha) in zip(r, t):
            row, col = coord
            m[row, col] = c
            a[row, col] = alpha

        #masked = np.ma.masked_where(m == -1, m)

        #cmap = plt.colormaps["RdYlGn"].copy()
        #cmap.set_bad((1,1,1,0))

        cmap = plt.colormaps["RdYlGn"]

        rgba = cmap(np.nan_to_num(m, nan=0))
        rgba[..., 3] = np.where(np.isnan(m), 0, a)
        
        ish = plt.imshow(
            rgba,
            origin="lower",
            cmap=cmap,
            interpolation="none"
        )

        plt.colorbar(ish, label="Survived percentage")
        plt.show()






    if False: #plot probabilities stuff (need to set do_analysis = True in simulation __init__)
        rp_u = [(i[1],i[2]) for i in res_of_p if i[0] == sim_config.URGENCY_U]
        rp_n = [(i[1],i[2]) for i in res_of_p if i[0] == sim_config.URGENCY_N]

        log_likelihood_u = sum(
            math.log(p) if result else math.log(1 - p)
            for p, result in rp_u
        )
        log_likelihood_n = sum(
            math.log(p) if result else math.log(1 - p)
            for p, result in rp_n
        )
        print(log_likelihood_u/len(rp_u), log_likelihood_n/len(rp_n))


        mean_p = sum(p for p, _ in rp_u) / len(rp_u)
        observed = sum(int(s) for _, s in rp_u) / len(rp_u)

        print("mean predicted:", mean_p)
        print("observed:", observed)


        data = defaultdict(list)

        for typ, dist, p in p_by_dist:
            data[typ].append((dist, p))

        data[0].sort()
        x1, y1 = zip(*data[1]) if data[1] else ([], [])
        x2, y2 = zip(*data[2]) if data[2] else ([], [])
        

        # plot
        plt.scatter(x1,y1, marker='o', label='Non-urgent')
        plt.scatter(x2, y2, marker='o', label='urgent')

        plt.xlabel("Distanz")
        plt.ylabel("p")
        plt.legend()
        plt.grid(True)
        plt.show()

    

    """
    profiler = cProfile.Profile()
    profiler.enable()

    sim_only()

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumtime")
    stats.print_stats(20)
    """
    
    if False:
        res = []
        for i in tqdm(range(1,10)):
            sim_config.END_DAYS=i
            r,t = sim_only()
            r = asdict(r.to_scalar())
            r["time"] = t/6
            res.append(r)

        df = pd.DataFrame(res)
        df[[
            "time",
            "death_rate",
            "not_admitted_rate",
            #"urgent_death_rate",
            "urgent_not_admitted_rate",
            #"normalized_admitted_distance",
            #"normalized_not_survived_distance",
            #"normalized_admitted_choice_rank",
            #"normalized_not_survived_choice_rank",
            #"normalized_unused_hospitals",
            #"small_summed",
            #"normalized_cost",
            #"reconstructed_fitness"
        ]].plot(kind='line', legend=True, figsize=(12,6))
        df[[
            "time",
            "death_rate",
            "not_admitted_rate",
            "urgent_death_rate",
            "urgent_not_admitted_rate",
            "normalized_admitted_distance",
            #"normalized_not_survived_distance",
            "normalized_admitted_choice_rank",
            "normalized_not_survived_choice_rank",
            "normalized_unused_hospitals",
            #"small_summed",
            "normalized_cost",
            #"reconstructed_fitness"
        ]].plot(kind='line', legend=True, figsize=(12,6))
        df[[
            "time",
            "death_rate",
            "not_admitted_rate",
            "urgent_death_rate",
            "urgent_not_admitted_rate",
            #"normalized_admitted_distance",
            #"normalized_not_survived_distance",
            "normalized_admitted_choice_rank",
            "normalized_not_survived_choice_rank",
            "normalized_unused_hospitals",
            #"small_summed",
            #"normalized_cost",
            #"reconstructed_fitness"
        ]].plot(kind='line', legend=True, figsize=(12,6))
        plt.ylabel("Value")
        plt.title("Simulation Records")
        plt.tight_layout()
        plt.show()
    
    

    #cProfile.run("sim_only()")
    pass



