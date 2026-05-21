from abc import ABC, abstractmethod
from ga.individual import Individual
from typing import Tuple
from tqdm import tqdm
import os
import copy

from concurrent.futures import ProcessPoolExecutor
from functools import partial

from Simulation import Simulation
from Simulation import SimConfig

from ga.individual import Individual


class Evaluator(ABC):

    def __init__(
            self,
            sim_config : SimConfig,
            cities : list[Tuple[int,int,int]],
            cities_matrix = None
        ):
        super().__init__()

        self.sim_config = sim_config
        self.cities = cities
        self.cities_matrix = cities_matrix



    @abstractmethod
    def evaluate(
        self,
        individuals : list[Individual]
    ) -> None:
        pass


class SerialEvaluator(Evaluator):

    def evaluate(
        self,
        individuals : list[Individual]
    ) -> None:
        pass


class ParallelEvaluator(Evaluator):

    def __init__(
            self,
            sim_config : SimConfig,
            cities : list[Tuple[int,int,int]],
            n_workers : int = 16,
            cities_matrix = None,
        ):
        super().__init__(sim_config, cities,cities_matrix=cities_matrix)

        self.workers = n_workers
        self.sim_config = sim_config

    @staticmethod
    def _evaluate_single(individual, simulation_config, cities,cities_matrix):
        simulation = Simulation(
            start_pos=individual.genome,
            sc=simulation_config,
            cities_list=cities.copy(),
            cities=cities_matrix
        )
        f = simulation.run()
        print(f)
        individual.fitness = f
        return individual

    def evaluate(
            self, 
            individuals: list[Individual]
        ) -> None:

        fn = partial(
            self._evaluate_single,
            simulation_config=self.sim_config,
            cities=self.cities,
            cities_matrix=self.cities_matrix
        )

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            results = list(tqdm(
                executor.map(fn, individuals),
                total=len(individuals)
            ))

        individuals[:] = results 