from ga.variation.basic import BasicMutationVariation, BasicCrossoverVariation
from ga.variation.classic import ClassicVariation
from ga.variation.evolutionary import EvolutionaryVariation
from ga.variation.microGA import MicroGAVariation
from ga.variation.base import VariationStrategy

__all__ = [
    "BasicMutationVariation",
    "BasicCrossoverVariation",
    "ClassicVariation",
    "EvolutionaryVariation",
    "MicroGAVariation"
]
