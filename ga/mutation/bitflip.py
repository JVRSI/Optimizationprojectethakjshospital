from ga.mutation.base import MutationStrategy


class BitFlipMutation(MutationStrategy):

    def __init__(self, probability=0.01):
        self.probability = probability

    def mutate(self, individual):
        pass