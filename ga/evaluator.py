from abc import ABC, abstractmethod
from ga.individual import Individual


class Evaluator(ABC):

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

    def __init__(self, simulation_config, workers=4):
        self.simulation_config = simulation_config
        self.workers = workers

    def evaluate(
        self,
        individuals : list[Individual]
    ) -> None:
        pass