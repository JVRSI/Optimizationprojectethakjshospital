from abc import ABC, abstractmethod


class CrossoverStrategy(ABC):

    @abstractmethod
    def crossover(self, parent1, parent2):
        pass