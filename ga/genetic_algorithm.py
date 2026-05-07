from ga.population import Population
from ga.individual import Individual


class GeneticAlgorithm:

    def __init__(
        self,
        population_size,
        genome_generator,
        selection,
        crossover,
        mutation,
        replacement,
        evaluator,
    ):

        self.population_size = population_size

        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.replacement = replacement
        self.evaluator = evaluator

        self.population = Population(
            [
                Individual(genome_generator())
                for _ in range(population_size)
            ]
        )

    def initialize(self):
        pass

    def step(self):
        pass

    def run(self, generations):
        pass

    def create_offspring(self, parents):
        pass

    def evaluate_population(self):
        pass