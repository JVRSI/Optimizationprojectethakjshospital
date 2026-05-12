from ga.individual import Individual
from statistics import mean


class Population:

    def __init__(
        self,
        individuals : list[Individual]
    ):
        self.individuals = individuals
        self.best_individual = None
        self.worst_individual = None
        self.average_fitness = None

    def sort_population(self) -> None:
        self.individuals = sorted(self.individuals, key=lambda x: x.fitness)

        self.best = self.individuals[-1]
        self.worst = self.individuals[0]
        self.average = mean(x.fitness for x in self.individuals)

    def best(self) -> Individual:
        if self.best_individual is None:
            self.sort_population()
        return self.best_individual
    
    def worst(self) -> Individual:
        if self.worst_individual is None:
            self.sort_population()
        return self.worst_individual
    
    def average(self) -> float:
        if self.average_fitness is None:
            self.sort_population()
        return self.average_fitness

    def size(self):
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)