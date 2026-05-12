from dataclasses import dataclass

@dataclass
class SimConfig:
    SEED = None
    END_DAYS = 20

    # Hospital capacities
    # Type 2 = large / major hospital
    # Type 1 = smaller regional hospital
    CAPACITYL = 100
    CAPACITYS = 2

    # Daily probability that one person needs hospital treatment.
    # These values are intentionally much lower than before because the
    # matrix contains the full Swiss population.
    SICK_RATE_U = 0.00002   # urgent cases per person per day
    SICK_RATE_N = 0.00008   # non-urgent cases per person per day

    # Average days in treatment
    PATIENT_DAYS_U = 3
    PATIENT_DAYS_N = 7

    # Urgency categories
    URGENCY_U = 2  # urgent
    URGENCY_N = 1  # non-urgent

    # Distance-based survival model.
    # One reduced matrix cell is approximately 1 km.
    # Therefore the distance penalties must be very small per grid cell.
    BASE_SURVIVAL_PROB_U = 0.995
    BASE_SURVIVAL_PROB_N = 0.999
    DISTANCE_PENALTY_U = 0.0015
    DISTANCE_PENALTY_N = 0.0002
    SURVIVAL_NOISE_STD_U = 0.01
    SURVIVAL_NOISE_STD_N = 0.003