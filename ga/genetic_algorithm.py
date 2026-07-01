import time
import heapq


from ga.population import Population
from ga.individual import Individual
from ga.config import GAConfig
from ga.selection.base import SelectionStrategy
from ga.variation.base import VariationStrategy
from ga.evaluator import Evaluator
from ga.genome_generator import GenomeGenerator, LoadFromFile
from ga.analysis.statistics import GAStatistics

import numpy as np





class GeneticAlgorithm:

    def __init__(
        self,
        config : GAConfig,
        genome_generator : GenomeGenerator,
        selection : SelectionStrategy,
        variation : VariationStrategy,
        evaluator : Evaluator,
        statistics : GAStatistics,
        rng : np.random.Generator = None,
    ):
        if not isinstance(config, GAConfig):
            raise TypeError(f"config must be instance of GAConfig, got {type(config)}")
        if not isinstance(genome_generator, GenomeGenerator):
            raise TypeError(f"genome_generator must be instance of GenomeGenerator, got {type(genome_generator)}")
        if not isinstance(selection, SelectionStrategy):
            raise TypeError(f"selection must be instance of Selection, got {type(selection)}")
        if not isinstance(variation, VariationStrategy):
            raise TypeError(f"crossover must be instance of Crossover, got {type(variation)}")
        if not isinstance(evaluator, Evaluator):
            raise TypeError(f"evaluator must be instance of Evaluator, got {type(evaluator)}")

        self.config = config

        self.statistics = statistics

        self.selection = selection
        self.variation = variation
        self.evaluator = evaluator


        self.genome_generator = genome_generator
        self.loadingFromFile = False
        if isinstance(self.genome_generator, LoadFromFile):
            self.loadingFromFile = True
        self.population = None
        self.time_initial = time.time()


    def initialize(self) -> None:
        """
        Initializes and evaluates generation 0

        returns time of simulation
        """

        if self.population is None and not self.loadingFromFile:
            self.population = Population(
                [
                    Individual(self.genome_generator())
                    for _ in range(self.config.initial_population_size)
                ]
            )
        elif not self.loadingFromFile:
            self.population.individuals.extend(
                [
                    Individual(self.genome_generator())
                    for _ in range(self.config.initial_population_size - self.population.size())
                ]
            )
        else:
            self.population = Population(
                [
                    Individual(genome)
                    for genome in iter(self.genome_generator, None)
                ]
            )




        # initial evaluation
        ta = time.time()
        self.evaluator.evaluate(
            self.population.individuals
        )
        te = time.time()

        self.population.clear_stats()

        return te - ta


    def step(self) -> float:
        """
        Updating one generation

        returns time of evaluating the simulation
        """

        #stuff is done inplace (hopefully)

        elites = []
        if self.config.n_elites > 0:
            best_is = heapq.nsmallest(self.config.n_elites, range(len(self.population.individuals)), key=lambda i : self.population.individuals[i].fitness)
            elites = [self.population.individuals[i].clone() for i in best_is]

        # select parents to construct offspring from
        self.selection.select(self.population)

        # create offspring
        self.variation.variate(self.population.individuals)

        # evaluate offspring
        ta = time.time()
        self.evaluator.evaluate(self.population.individuals)
        te = time.time()

        self.population.clear_stats()

        self.population.individuals.extend(elites)

        #+ statistics
        return te - ta


    def run(self):
        """
        Runs genetic algorithm

        returns best individual and exit status {"max_iterations", "max_time", "converged"}
        """
        generations = self.config.n_generations

        exit_status = "max_iterations"


        #initialize
        ta = time.time()
        ts = self.initialize()
        te = time.time()

        #statistics of generation 0
        if self.config.collect_performance_data:
            self.statistics.record(
                generation=0,
                population=self.population,
                time_creating_offspring=(te-ta-ts),
                time_simulation_total=ts,
            )

        for generation in range(1, generations+1):
            tc = time.time()
            print(f"remaining time: {(self.config.max_time_to_run_s - (tc - self.time_initial))/60} min")
            
            if tc - self.time_initial > self.config.max_time_to_run_s:
                print("+---------------------------------------------------------------------+")
                print("| Stopped because out of time                                         |")
                print("+---------------------------------------------------------------------+")
                exit_status = "max_time"
                break

            # check convergence
            #if self.statistics.steps_from_last_improvement >= self.config.n_steps_of_no_improvement_to_converge:
            if self.statistics.relative_improvement(window=self.config.n_steps_of_no_improvement_to_converge, std_threshold=self.config.std_threshold) < self.config.slop_threshold:
                #random restart
                if self.config.do_random_restarts:
                    print("+---------------------------------------------------------------------+")
                    print("| Doing random restart                                                |")
                    print("+---------------------------------------------------------------------+")

                    self.population.delete_worst_individuals(self.config.n_best_to_keep)

                    self.statistics.steps_from_last_improvement = 0
                    self.statistics.current_best = 100

                    ta = time.time()
                    ts = self.initialize()
                    te = time.time()

                    #statistics of restart
                    if self.config.collect_performance_data:
                        self.statistics.record(
                            generation=generation,
                            population=self.population,
                            time_creating_offspring=(te-ta-ts),
                            time_simulation_total=ts,
                            currently_random_restart=True,
                        )
                    continue
                else:
                    print("+---------------------------------------------------------------------+")
                    print("| Stopped because converged                                           |")
                    print("+---------------------------------------------------------------------+")
                    exit_status = "converged"
                    break

            print("-----------------------------------------------------------------------")
            print(f"Generation {generation}/{generations}")
            ta = time.time()
            ts = self.step()
            te = time.time()
            
            if self.config.collect_performance_data:
                self.statistics.record(
                    generation=generation,
                    population=self.population,
                    time_simulation_total=ts,
                    time_creating_offspring=(te-ta-ts),
                )        


        best_individual = self.population.best()

        return best_individual, exit_status
