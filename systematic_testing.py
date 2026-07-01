import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
import json
import sys
import argparse
import time
import traceback
from dataclasses import replace



from paths import DATA_DIR, RUNS_DIR, MATRIX_PATH, RUNS_LOCAL

from ga import GAConfig, GeneticAlgorithm, BasicGenerator, GravityGenerator, ParallelEvaluator
from ga.genome_generator import LoadFromFile
from ga.selection import RouletteSelection, TruncateSelection
from ga.variation import EvolutionaryVariation, MicroGAVariation, ClassicVariation
from ga.analysis.statistics import GAStatistics
from gaFactory import GAFactory

from testSim import matrix_to_city_dataframe, load_population_matrix



from Simulation import Simulation, SimConfig
from Simulation.entities import City


# runs configs
from systematic_runs_configs.r8 import runs
runs_conf = "r9"

#r6 skip_runs = [0,1,8,16,17]
skip_runs = []

save_stuff = True

if not save_stuff:
    print("\033[38;5;160mATTENTION\033[0m Nothing is saved in this run")


with open(DATA_DIR / "gov_data" / "cities_list_reduced_from_root_rounded_with_coverage.pkl", "rb") as f:
    cities_list = pickle.load(f)


parser = argparse.ArgumentParser()
parser.add_argument("-w", "--n-workers", type=int, default=16)
args = parser.parse_args()

size = (219,345)
seed = 1782458519 #int(time.time()) #r6 seed: 1780652528 # r8_0 seed: 1782458519

n_workers = args.n_workers
record_history_of_best_and_worst = True
runs_dir = RUNS_DIR / f"0_systematic_{runs_conf}"


cities_matrix = load_population_matrix(MATRIX_PATH)

runs = runs(size=size, record_history_of_best_and_worst=record_history_of_best_and_worst, cities_matrix=cities_matrix, seed=seed)




###############################################################################################
#Sim config
sim_config = SimConfig(
    SEED=seed,
    END_DAYS=100,
    CAPACITYL_N = 100,
    CAPACITYL_U = 50,
    CAPACITYS_N = 0,
    CAPACITYS_U = 20,
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
################################################################################################




def main():
    #always same: sim_config, evaluator
    #loop: new GA stats, rng, ... per run

    failed_runs = []

    seed_seq = np.random.SeedSequence(seed)
    seeds = seed_seq.spawn(4)

    date_start = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if runs_dir.exists():
        cur_i = max(
            (
                int(p.name.split("_")[0])
                for p in runs_dir.iterdir()
                if p.is_dir()
                and "_" in p.name
                and p.name.split("_")[0].isdigit()
            ),
            default=-1
        ) + 1
    else:
        cur_i = 0
    
    
    


    if save_stuff:
        runs_dir.mkdir(exist_ok=True)
        with open(runs_dir / f"{cur_i}_seeds_{date_start}.json", "w") as f:
            json.dump({
                "master_entropy": seed_seq.state,
                "seeds_keys":[s.state for s in seeds],
            }, f, indent=4)



    for i, run in enumerate(runs):
        if i in skip_runs:
            print(f"*******************************+--------------{''             }------+***********************************")
            print(f"*******************************| Skipping  {(i+1):3}/{len(runs):<3}  |***********************************")
            print(f"*******************************+--------------{''             }------+***********************************")
            continue
        try: 
            print(f"*******************************+-------------{''             }-------+***********************************")
            print(f"*******************************|      Run  {(i+1):3}/{len(runs):<3}  |***********************************")
            print(f"*******************************+-------------{''             }-------+***********************************")
            ga_config = run.ga_config
            ga_stats = GAStatistics(record_individual_history=ga_config.record_individual_history)
            if run.sim_config is not None:
                sc = run.sim_config
            else:
                sc = sim_config
            evaluator = ParallelEvaluator(sim_config=sc,cities=cities_list,n_workers=n_workers, rng=np.random.default_rng(seeds[0]),record_individual_history=ga_config.record_individual_history)
         

            genome_generator = run.generator_factory(ga_config,seeds[1])
            selector = run.selector_factory(ga_config.n_parents,seeds[2])
            variator = run.variator_factory(ga_config,seeds[3])


            ga = GeneticAlgorithm(
                ga_config,
                genome_generator=genome_generator,
                selection=selector,
                variation=variator,
                evaluator=evaluator,
                statistics=ga_stats,
            )
            best, status = ga.run()
            last_population_dict = ga.population.to_dict()


            print("-------------------------------------------------------------------------------")
            print(status)


            if save_stuff:
                # save run result, config and stats (in a new folder within runs)
                dir_name = f"{cur_i}_{i}_{selector.__class__.__name__}_{variator.__class__.__name__}_{genome_generator.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{evaluator.__class__.__name__}"
                dir_path = runs_dir / dir_name

                # stats and plots
                if ga_config.collect_performance_data:
                    ga_stats.save_csv(dir_path, ga_config.plot_images, sc)

                # config
                ga_config_dict = asdict(ga_config)
                with open(dir_path / "ga_config.json", "w") as f:
                    json.dump(ga_config_dict, f, indent=4)

                sim_config_dict = asdict(sc)
                with open(dir_path / "sim_config.json", "w") as f:
                    json.dump(sim_config_dict, f, indent=4)

                # result
                best_dict = best.to_dict()
                with open(dir_path / "best.json", "w") as f:
                    json.dump(best_dict, f, indent=4)

                with open(dir_path / "last_generation.json", "w") as f:
                    json.dump(last_population_dict, f, indent=4)
        
            #"""
        except KeyboardInterrupt:
            print("+-------------------------------------------------------------------------+")
            print("| Stopped execution, doing clean up and save stuff                        |")
            print("+-------------------------------------------------------------------------+")
            break
        except Exception as e:
            print("###########################################################################")
            print(f"Run {i} failed")
            print(run)
            print("---------------------------------------------------------------------------")
            print(e)
            print("---------------------------")
            print(traceback.format_exc())
            print("###########################################################################")
            failed_runs.append([i,str(run),str(e),traceback.format_exc()])
            #"""
        
    if save_stuff:
        with open(runs_dir / f"{cur_i}_failed_runs_{date_start}.json", "w") as f:
            json.dump(failed_runs, f, indent=4)
    


def cleanup_runs():
    #like main, but with different seed per run, and otherwise more control and directly here generated run configs

    failed_runs = []

    

    date_start = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    runs_dir_cleanup = RUNS_DIR / "0_cleanup"

    if runs_dir_cleanup.exists():
        cur_i = max(
            (
                int(p.name.split("_")[0])
                for p in runs_dir_cleanup.iterdir()
                if p.is_dir()
                and "_" in p.name
                and p.name.split("_")[0].isdigit()
            ),
            default=-1
        ) + 1
    else:
        cur_i = 0
    
    

    if save_stuff:
        runs_dir_cleanup.mkdir(exist_ok=True)
        


    ##############################################################################################################################
    # Configs
    ##############################################################################################################################

    folder_of_best =  RUNS_DIR / f"0_systematic_r6" / "2_5_TruncateSelection_ClassicVariation_GravityGenerator_2026-06-08_22-44-47_ParallelEvaluator"

    seeds = [
        # 0, MC
        (lambda s: (s, np.random.SeedSequence(s).spawn(4)))(1780652528), #r6, 2,
        (lambda s: (s, np.random.SeedSequence(s).spawn(4)))(int(time.time())),
        (lambda s: (s, np.random.SeedSequence(s).spawn(4)))(int(time.time())),
        (lambda s: (s, np.random.SeedSequence(s).spawn(4)))(int(time.time())),
    ]

    gc = GAConfig(
        n_generations=1000,
        initial_population_size=100,
        population_size=100,
        genome_size= size,
        mean_hospital_large=29,
        mean_hospital_small=101,
        random_amount = False,
        collect_performance_data=True,
        plot_images=True,
        record_individual_history=record_history_of_best_and_worst,
        n_parents=10,
        n_elites=2,
        n_hospital_types=2, #not everywhere is support for variable hospital types
        mutation_strategy="wandering", #{wandering, mutable_wandering, single_point, single_point_equal_opportunity, teleport, wandering_teleport}
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
    def sc(seed_sc): 
        return SimConfig(
            SEED=seed_sc,
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
    runs_cleanup = [
        #-----------------------------------------------------------------------------------------------------------
        # MC 2h
        GAFactory(
            ga_config = replace(gc,
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
            sim_config= lambda seed: sc(seed_sc=seed),
        ),
        #-----------------------------------------------------------------------------------------------------------
        # Improve on best
        GAFactory(
            ga_config = replace(gc,
                mutation_strategy="wandering",
            ),
            generator_factory = lambda config, seed: LoadFromFile(rng=np.random.default_rng(seed),config=config, folder_path=folder_of_best),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: EvolutionaryVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="LoadFromFile",
            selector_name="TruncateSelection",
            variator_name="EvolutionaryVariation",
            idea="",
            sim_config= lambda seed: sc(seed_sc=seed),
        ),
        #-----------------------------------------------------------------------------------------------------------
        # Improve on best smaller sigma
        GAFactory(
            ga_config = replace(gc,
                mutation_strategy="wandering",
                wandering_mutation_sigma=5,
            ),
            generator_factory = lambda config, seed: LoadFromFile(rng=np.random.default_rng(seed),config=config, folder_path=folder_of_best),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: EvolutionaryVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="LoadFromFile",
            selector_name="TruncateSelection",
            variator_name="EvolutionaryVariation",
            idea="",
            sim_config= lambda seed: sc(seed_sc=seed),
        ),
        #-----------------------------------------------------------------------------------------------------------
        # Improve on best compare
        GAFactory(
            ga_config = replace(gc,
                probability_of_mutation=0.8694,
                probability_of_crossover=0.6554,
                mutation_strategy="wandering",
            ),
            generator_factory = lambda config, seed: LoadFromFile(rng=np.random.default_rng(seed),config=config, folder_path=folder_of_best),
            selector_factory = lambda n_parents, seed: TruncateSelection(n_parents=n_parents,rng=np.random.default_rng(seed)),
            variator_factory = lambda config, seed: ClassicVariation(ga_config=config,rng=np.random.default_rng(seed)),
            generator_name="LoadFromFile",
            selector_name="TruncateSelection",
            variator_name="ClassicVariation",
            idea="",
            sim_config= lambda seed: sc(seed_sc=seed),
        ),

    ]

    

    print("################################+--------------------+###################################")
    print("################################|    Cleanup runs    |###################################")
    print("################################+--------------------+###################################")
    for i, run in enumerate(runs_cleanup):
        
        if i in skip_runs:
            print(f"*******************************+--------------{''                     }------+***********************************")
            print(f"*******************************| Skipping  {(i+1):3}/{len(runs_cleanup):<3}  |***********************************")
            print(f"*******************************+--------------{''                     }------+***********************************")
            continue
        try: 
            print(f"*******************************+-------------{''                     }-------+***********************************")
            print(f"*******************************|      Run  {(i+1):3}/{len(runs_cleanup):<3}  |***********************************")
            print(f"*******************************+-------------{''                     }-------+***********************************")
            ga_config = run.ga_config
            ga_stats = GAStatistics(record_individual_history=ga_config.record_individual_history)
            if run.sim_config is not None:
                sc = run.sim_config(seeds[i][0])
            else:
                sc = sim_config
            evaluator = ParallelEvaluator(sim_config=sc,cities=cities_list,n_workers=n_workers, rng=np.random.default_rng(seeds[i][1][0]),record_individual_history=ga_config.record_individual_history)
         

            genome_generator = run.generator_factory(ga_config,seeds[i][1][1])
            selector = run.selector_factory(ga_config.n_parents,seeds[i][1][2])
            variator = run.variator_factory(ga_config,seeds[i][1][3])


            ga = GeneticAlgorithm(
                ga_config,
                genome_generator=genome_generator,
                selection=selector,
                variation=variator,
                evaluator=evaluator,
                statistics=ga_stats,
            )
            best, status = ga.run()
            last_population_dict = ga.population.to_dict()


            print("-------------------------------------------------------------------------------")
            print(status)


            if save_stuff:
                # save run result, config and stats (in a new folder within runs)
                dir_name = f"{cur_i}_{i}_{selector.__class__.__name__}_{variator.__class__.__name__}_{genome_generator.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{evaluator.__class__.__name__}"
                dir_path = runs_dir_cleanup / dir_name

                # stats and plots
                if ga_config.collect_performance_data:
                    ga_stats.save_csv(dir_path, ga_config.plot_images, sc)

                # config
                ga_config_dict = asdict(ga_config)
                with open(dir_path / "ga_config.json", "w") as f:
                    json.dump(ga_config_dict, f, indent=4)

                sim_config_dict = asdict(sc)
                with open(dir_path / "sim_config.json", "w") as f:
                    json.dump(sim_config_dict, f, indent=4)

                # result
                best_dict = best.to_dict()
                with open(dir_path / "best.json", "w") as f:
                    json.dump(best_dict, f, indent=4)

                with open(dir_path / "last_generation.json", "w") as f:
                    json.dump(last_population_dict, f, indent=4)
                
                with open(runs_dir_cleanup / f"{cur_i}_seeds_{date_start}.json", "w") as f:
                    json.dump({
                        "master_entropy": seeds[i][0],
                    }, f, indent=4)
                with open(runs_dir_cleanup / f"{cur_i}_seeds_{date_start}.json", "w") as f:
                    json.dump({
                        "master_entropy": seeds[i][0],
                    }, f, indent=4)
        
            #"""
        except KeyboardInterrupt:
            print("+-------------------------------------------------------------------------+")
            print("| Stopped execution, doing clean up and save stuff                        |")
            print("+-------------------------------------------------------------------------+")
            break
        except Exception as e:
            print("###########################################################################")
            print(f"Run {i} failed")
            print(run)
            print("---------------------------------------------------------------------------")
            print(e)
            print("---------------------------")
            print(traceback.format_exc())
            print("###########################################################################")
            failed_runs.append([i,str(run),str(e),traceback.format_exc()])
            #"""
        
    if save_stuff:
        with open(runs_dir_cleanup / f"{cur_i}_failed_runs_{date_start}.json", "w") as f:
            json.dump(failed_runs, f, indent=4)
    



if __name__ == "__main__":


    cleanup_runs()

    main()











