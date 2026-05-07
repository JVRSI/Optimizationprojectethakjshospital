from abc import ABC, abstractmethod


class Evaluator(ABC):

    @abstractmethod
    def evaluate(self, individuals):
        pass


class SerialEvaluator(Evaluator):

    def __init__(self, simulation):
        self.simulation = simulation

    def evaluate(self, individuals):
        pass


class ParallelEvaluator(Evaluator):

    def __init__(self, simulation_config, workers=4):
        self.simulation_config = simulation_config
        self.workers = workers

    def evaluate(self, individuals):
        pass