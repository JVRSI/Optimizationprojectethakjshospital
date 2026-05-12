import heapq

# agents who get send to hospital
class Patient:
    def __init__(self, days, home, urgency):
        self.days = days
        self.home = home
        self.urgency = urgency

# Object for the cells
class City:
    def __init__(self, id, btot, inHospital, hospitals):
        self.city_id = id
        self.btot = btot
        self.hospitals_sorted = hospitals
        self.in_hospital = inHospital 
    has_hospital = False
    hospital_ob = None


class Hospital:
    capacity = 0

    def __init__(self, type, id, location, sc):
        self.hos_id = id
        self.type = type
        self.location = location
        self.patientqueue = []
        self.current_load = 0
        self.patient_counter = 0
        self.sc = sc
        if type == 2: # Large hospital
            self.capacity = sc.CAPACITYL
            self.cost = sc.COSTL
        elif type == 1: # Small hospital
            self.capacity = sc.CAPACITYS
            self.cost = sc.COSTS

    def can_treat(self, patient):
        if self.type == 2:
            return True
        return patient.urgency == self.sc.URGENCY_U

    def add_patient(self, patient):
        if self.current_load < self.capacity:
            heapq.heappush(self.patientqueue, (patient.days, self.patient_counter, patient))
            self.patient_counter += 1
            self.current_load += 1
            return True
        return False

    def treat_next(self, current_step):
        if self.patientqueue and self.patientqueue[0][0] == current_step:
            _, _, patient = heapq.heappop(self.patientqueue)
            self.current_load -= 1
            return patient
        return None