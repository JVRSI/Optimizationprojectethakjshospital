from dataclasses import dataclass

@dataclass
class SimConfig:
    SEED : int = None
    END_DAYS : int = 20

    # Hospital capacities
    # Type 2 = large / major hospital
    # Type 1 = smaller regional hospital
    CAPACITYL_N : int = 100
    CAPACITYL_U: int = 50
    CAPACITYS_N  : int = 0
    CAPACITYS_U  : int = 10
    COSTL : float = 10
    COSTS :float = 5
    TOTALCOST : float = 300

    # Daily probability that one person needs hospital treatment.
    # These values are intentionally much lower than before because the
    # matrix contains the full Swiss population.
    SICK_RATE_U : float = 0.00002   # urgent cases per person per day
    SICK_RATE_N : float = 0.00008   # non-urgent cases per person per day

    # Average days in treatment
    PATIENT_DAYS_U : int = 3
    PATIENT_DAYS_N : int = 7

    # Urgency categories
    URGENCY_U : int = 2  # urgent
    URGENCY_N : int = 1  # non-urgent

    # Distance-based survival model.
    # One reduced matrix cell is approximately 1 km.
    # Therefore the distance penalties must be very small per grid cell.
    BASE_SURVIVAL_PROB_U : float = 0.995
    BASE_SURVIVAL_PROB_N : float = 0.999
    DISTANCE_PENALTY_U : float = 0.0015
    DISTANCE_PENALTY_N : float = 0.0002
    SURVIVAL_NOISE_STD_U : float = 0.01
    SURVIVAL_NOISE_STD_N : float = 0.003


    #fitness calculation
    death_rate_factor: float = 0.30
    not_admitted_rate_factor: float = 0.30
    urgent_death_rate_factor: float = 0.15
    urgent_not_admitted_rate_factor: float = 0.05
    normalized_admitted_distance_factor: float = 0.05
    normalized_not_survived_distance_factor: float = 0.03
    normalized_admitted_choice_rank_factor: float = 0.04
    normalized_not_survived_choice_rank_factor: float = 0.03
    normalized_unused_hospitals_factor: float = 0.03
    normalized_cost_factor: float = 0.12
    bad_coverage_factor : float = 0.5

