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


from paths import DATA_DIR, RUNS_DIR, MATRIX_PATH, RUNS_LOCAL

from ga import GAConfig, GeneticAlgorithm, BasicGenerator, GravityGenerator, ParallelEvaluator
from ga.selection import RouletteSelection
from ga.variation import EvolutionaryVariation, MicroGAVariation, ClassicVariation
from ga.analysis.statistics import GAStatistics
from gaFactory import GAFactory

from testSim import matrix_to_city_dataframe, load_population_matrix



from Simulation import Simulation, SimConfig
from Simulation.entities import City


# runs configs
from systematic_runs_configs.r7 import runs
runs_conf = "r7"

#r6 skip_runs = [0,1,8,16,17]
skip_runs = [0,1]

save_stuff = True

if not save_stuff:
    print("\033[38;5;160mATTENTION\033[0m Nothing is saved in this run")


with open(DATA_DIR / "gov_data" / "cities_list_reduced_from_root_rounded_with_coverage.pkl", "rb") as f:
    cities_list = pickle.load(f)


parser = argparse.ArgumentParser()
parser.add_argument("-w", "--n-workers", type=int, default=16)
args = parser.parse_args()

size = (219,345)
seed = int(time.time()) #r6 seed: 1780652528

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
    
    



if __name__ == "__main__":

    main()











