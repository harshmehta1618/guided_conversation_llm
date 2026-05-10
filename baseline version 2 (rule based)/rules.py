class Rules:
    def __init__(self, persona:dict, task:dict):
        self._persona = persona
        self.task = task
        self.process()

    def generate_task(self, ref_task:str):
        convo_len = len(self.task)
        self.task[f'{convo_len+1}'] = ref_task
    
    def sleep_blood_pressure(self):
        sleep_hr = self._persona['Average_Sleeping_Hours']
        bp = self._persona['Has_High_Blood_Pressure']
        if sleep_hr != -1 and bp != -1:
            if sleep_hr < 8 and bp == 1:
                self.generate_task("Try to get more sleep to reduce BP")
    
    def exercise_weight(self):
        exercise = self._persona['Daily_Exercise']
        weight = self._persona['Weight']
        if exercise != -1 and weight != -1:
            if exercise != 1 and weight > 90:
                self.generate_task("Try to do more exercise to reduce weight")

    def tremors_lexapro(self):
        tremors = self._persona.get('Has_Tremors', -1)
        meds = self._persona.get('Medications', [])
        if tremors == 1 and 'Lexapro' in meds:
            self.generate_task("Monitor tremors as a side effect of Lexapro; consult doctor if they worsen")

    def white_coat_bp(self):
        bp = self._persona.get('Has_High_Blood_Pressure', -1)
        white_coat = self._persona.get('White_Coat_Hypertension', -1)
        if bp == 1 and white_coat == 1:
            self.generate_task("Monitor blood pressure at home to confirm white coat hypertension")

    def asthma_cough(self):
        asthma = self._persona.get('Has_Asthma', -1)
        cough = self._persona.get('Cough_With_Phlegm', -1)
        if asthma == 1 or cough == 1:
            self.generate_task("Avoid asthma triggers; use prescribed inhaler if needed; seek medical attention for severe cough")

    def nausea_vomiting(self):
        nausea = self._persona.get('Has_Nausea_Vomiting', -1)
        if nausea == 1:
            self.generate_task("Take prescribed anti-nausea medication like Zofran; stay hydrated and avoid solid foods temporarily")

    def stomach_pain_meds(self):
        stomach_pain = self._persona.get('Has_Stomach_Pain', -1)
        meds = self._persona.get('Medications', [])
        if stomach_pain == 1 and any(med in ['Aleve', 'NSAIDs'] for med in meds):
            self.generate_task("Avoid NSAIDs like Aleve; use Tylenol or consult doctor for stomach pain relief")

    def mouth_sores_dentures(self):
        sores = self._persona.get('Has_Mouth_Sores', -1)
        dentures = self._persona.get('Wears_Dentures', -1)
        if sores == 1 and dentures == 1:
            self.generate_task("Schedule dental check for mouth sores related to dentures")

    def shoulder_injury(self):
        injury = self._persona.get('Has_Shoulder_Injury', -1)
        if injury == 1:
            self.generate_task("Consider physical therapy for shoulder injury; follow up on rotator cuff if pain returns")

    def degenerative_changes(self):
        changes = self._persona.get('Degenerative_Changes', -1)
        if changes == 1:
            self.generate_task("Schedule regular check-ups for degenerative changes in neck and shoulder")

    def pneumonia_history(self):
        pneumonia = self._persona.get('Has_Pneumonia_History', -1)
        if pneumonia == 1:
            self.generate_task("Monitor for respiratory symptoms; get vaccinated against pneumonia if not done")

    def process(self):
        self.sleep_blood_pressure()
        self.exercise_weight()
        self.tremors_lexapro()
        self.white_coat_bp()
        self.asthma_cough()
        self.nausea_vomiting()
        self.stomach_pain_meds()
        self.mouth_sores_dentures()
        self.shoulder_injury()
        self.degenerative_changes()
        self.pneumonia_history()
        return self.task



    
