from dataclasses import dataclass
from typing import Tuple

@dataclass
class GAConfig:
    population_size: int = 200

    genome_size: Tuple[float, float] = (10, 20) # (height, width)
    mean_hospital_large: int = 30
    mean_hospital_small: float = 50

    collect_performance_data: bool = True # to safe to csv and plot
    plot_images : bool = True # to also create nice plots (only plots if data is collected)

    n_parents = 5

