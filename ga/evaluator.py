from abc import ABC, abstractmethod
from ga.individual import Individual
from typing import Tuple
from tqdm import tqdm
import os
import copy
import numpy as np

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
            cities_matrix = None,
            rng = None,
            record_individual_history:bool = False,
        ):
        super().__init__()

        self.sim_config = sim_config
        self.cities = cities
        self.rng = rng
        self.rih = record_individual_history
        #self.cities_matrix = cities_matrix



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
    @staticmethod
    def _wrapped(item, fn):
        i, ind = item
        return i, fn(ind)

    def __init__(
            self,
            sim_config : SimConfig,
            cities : list[Tuple[int,int,int]],
            n_workers : int = 16,
            rng = None,
            cities_matrix = None,
            record_individual_history : bool = False
        ):
        super().__init__(sim_config, cities,cities_matrix=cities_matrix, rng = rng, record_individual_history=record_individual_history)
        self.workers = n_workers

    @staticmethod
    def _evaluate_single(individual:Individual, simulation_config, cities, rng, rih):
        simulation = Simulation(
            start_pos=individual.genome,
            sc=simulation_config,
            cities_list=cities.copy(),
            rng = rng,
            #cities=cities_matrix,
        )
        individual.fitness = simulation.run(log=False)
        if rih:
            individual.sim_records = simulation.get_result().to_scalar()
        return individual

    def evaluate(self, individuals: list[Individual]) -> None:
        fn = partial(
            self._evaluate_single,
            simulation_config=self.sim_config,
            cities=self.cities,
            rng=np.random.default_rng(self.rng.integers()),
            rih=self.rih
        )

        # split
        to_run = [(i, ind) for i, ind in enumerate(individuals) if ind.fitness is None]


        # calculate new fitness values
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            computed = list(tqdm(
                executor.map(partial(ParallelEvaluator._wrapped, fn=fn), to_run),
                total=len(to_run)
            ))

        # assign new results
        for i, ind in computed:
            individuals[i] = ind
