from abc import ABC, abstractmethod
from ga.individual import Individual
from typing import Tuple

from concurrent.futures import ProcessPoolExecutor
from functools import partial

from Simulation import Simulation
from Simulation import SimConfig

from ga.individual import Individual


class Evaluator(ABC):

    def __init__(
            self,
            sim_config : SimConfig,
            cities : list[Tuple[int,int,int]]
        ):
        super().__init__()

        self.sim_config = sim_config
        self.cities = cities



    @abstractmethod
    def evaluate(
        self,
        individuals : list[Individual]
    ) -> None:
        pass


class SerialEvaluator(Evaluator):

    def __init__(self, simulation):
        self.simulation = simulation

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
            n_workers : int = 16
        ):
        super().__init__(sim_config, cities)

        self.workers = n_workers

    @staticmethod
    def _evaluate_single(individual, simulation_config, cities):
        simulation = Simulation(
            start_pos=individual.genome,
            sc=simulation_config,
            cities_list=cities.copy()
        )
        individual.fitness = simulation.run()
        return individual

    def evaluate(
            self, 
            individuals: list[Individual]
        ) -> None:

        fn = partial(
            self._evaluate_single,
            simulation_config=self.simulation_config,
            cities=self.cities
        )

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            results = list(executor.map(fn, individuals))

        individuals[:] = results 