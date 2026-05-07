from abc import ABC, abstractmethod


class ReplacementStrategy(ABC):

    @abstractmethod
    def replace(
        self,
        old_population,
        offspring,
    ):
        pass