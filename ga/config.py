from dataclasses import dataclass
from typing import Tuple

@dataclass
class GAConfig:
    population_size: int = 200

    genome_size: Tuple[float, float] = (10, 20) # (height, width)
    mean_hospital_large: int = 30
    mean_hospital_small: float = 50

    n_parents = 5

