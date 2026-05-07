class Population:

    def __init__(self, individuals):
        self.individuals = individuals

    def best(self):
        pass

    def worst(self):
        pass

    def average_fitness(self):
        pass

    def size(self):
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)