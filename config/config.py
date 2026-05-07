from dataclasses import dataclass
from ga.config import GAConfig

@dataclass
class Config:
    ga: GAConfig
    #sim: SimulationConfig