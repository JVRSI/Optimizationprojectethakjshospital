import heapq

# agents who get send to hospital
class Patient:
    __slots__ = ("days", "home", "urgency")
    def __init__(self, days, home, urgency):
        self.days = days
        self.home = home
        self.urgency = urgency

# Object for the cells
class City:
    __slots__ = ("city_id", "btot", "hospitals_sorted", "in_hospital", "location")
    def __init__(self, id, btot, inHospital, hospitals, location=(-10,-100)):
        self.city_id = id
        self.btot = btot
        self.hospitals_sorted = hospitals
        self.in_hospital = inHospital 
        self.location = location
    has_hospital = False
    hospital_ob = None


class Hospital:
    __slots__ = (
        "hos_id", "type","location","patientqueue","urgent_load","nonurgent_load","capacity_urgent", "capacity_nonurgent","patient_counter","sc","cost"
    )
    def __init__(self, type, id, location, sc):
        self.hos_id = id
        self.type = type
        self.location = location
        self.patientqueue = []
        self.urgent_load = 0
        self.nonurgent_load = 0
        self.patient_counter = 0
        self.capacity_urgent = 0
        self.capacity_nonurgent = 0
        self.cost = 0
        self.sc = sc
        if type == 2: # Large hospital
            self.capacity_urgent = sc.CAPACITYL_U
            self.capacity_nonurgent = sc.CAPACITYL_N
            self.cost = sc.COSTL
        elif type == 1: # Small hospital
            self.capacity_urgent = sc.CAPACITYS_U
            self.capacity_nonurgent = sc.CAPACITYS_N
            self.cost = sc.COSTS
    
    def can_treat(self, patient):
        if patient.urgency == self.sc.URGENCY_U:
            return self.capacity_urgent-self.urgent_load > 0
        if patient.urgency == self.sc.URGENCY_N:
            return self.capacity_nonurgent-self.nonurgent_load > 0
        return False
    
    def add_patient(self, patient):
        if patient.urgency == self.sc.URGENCY_U:
            if self.urgent_load >= self.capacity_urgent:
                return False
            self.urgent_load += 1
        elif patient.urgency == self.sc.URGENCY_N:
            if self.nonurgent_load >= self.capacity_nonurgent:
                return False
            self.nonurgent_load += 1
        else:
            return False

        heapq.heappush(self.patientqueue, (patient.days, self.patient_counter, patient))
        self.patient_counter += 1
        return True

    def treat_next(self, current_step):
        if self.patientqueue and self.patientqueue[0][0] == current_step:
            _, _, patient = heapq.heappop(self.patientqueue)
            if patient.urgency == self.sc.URGENCY_U:
                self.urgent_load -= 1
            else:
                self.nonurgent_load -= 1
            return patient
        return None