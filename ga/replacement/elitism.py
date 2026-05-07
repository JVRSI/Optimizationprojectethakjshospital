from ga.replacement.base import ReplacementStrategy


class ElitismReplacement(ReplacementStrategy):

    def __init__(self, elite_count=1):
        self.elite_count = elite_count

    def replace(
        self,
        old_population,
        offspring,
    ):
        pass