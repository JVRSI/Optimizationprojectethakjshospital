from ga.mutation.base import MutationStrategy


class GaussianMutation(MutationStrategy):

    def __init__(self, sigma=0.1, probability=0.1):
        self.sigma = sigma
        self.probability = probability

    def mutate(self, individual):
        pass