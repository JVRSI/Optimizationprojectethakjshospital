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
        self.hospitals = hospitals
        self.in_hospital = inHospital 
    has_hospital = False
    hospital_ob = None


class Hospital:
    def __init__(self, id, location, patientqueue):
        self.hos_id = id
        self.location = location
        self.patientqueue = patientqueue