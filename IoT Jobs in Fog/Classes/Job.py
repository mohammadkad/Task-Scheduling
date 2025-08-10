'''
1404-05-15, Completed
'''
class Job:
    def __init__(self, job_id):
        self.job_id = job_id
        '''
        1404-05-11
        NUM_SENSORS//2 > 21//2 > 10
        random.randint(3, NUM_SENSORS//2), random.randint(3, 10) - This generates a random integer between 3 and 10, inclusive
        Sample: 3, 4, 5, 6, 7, 8, 9, or 10


        random.sample(range(NUM_SENSORS), random.randint(3, NUM_SENSORS//2)) > random.sample(population, k)
        a list of k distinct numbers from 0 to 20.
        Sample: [20, 1, 12, 17, 8], [16, 6, 20, 15, 19], [4, 8, 3, 17, 11, 5, 20, 9, 0, 18]
        '''
        self.sensors = random.sample(range(NUM_SENSORS), random.randint(3, NUM_SENSORS//2))
        
        '''
        1404-05-12
        random.uniform(a, b), MIN_vCPU_PER_SENSOR = 0.01, MAX_vCPU_PER_SENSOR = 0.1
        This function returns a random floating-point number x such that a <= x <= b
        Sample: 0.0831867797584396
        '''
        self.vcpu_per_sensor = random.uniform(MIN_vCPU_PER_SENSOR, MAX_vCPU_PER_SENSOR)
        self.mem_per_sensor = random.uniform(MIN_MEM_PER_SENSOR, MAX_MEM_PER_SENSOR)
        
        self.vcpu_fl_server = vCPU_PER_FL_SERVER
        self.mem_fl_server = MEM_PER_FL_SERVER
        self.epochs = FL_EPOCHS
        self.model_size = MODEL_SIZE
        self.training_freq = FL_TRAINING_FREQ
        
        '''
        1404-05-14
        creates a dictionary
        keys are the sensor IDs and the values are the corresponding workload amounts.
        Sample : [10, 9, 17, 4, 2, 7, 20, 11, 1]
        {10: 82.21759472584355,  9: 80.87941444744027,  17: 480.7214997672905,  4: 253.56947236504593,2: 222.84119630022616,
          7: 171.85296078505777,  20: 417.86771945341377,  11: 116.47288856759488,  1: 326.7232437094912}
        '''
        self.workloads = {s: random.uniform(50, 500) for s in self.sensors}  # MB/day
        
    '''
    1404-05-15
    return a list of 28 (21 sensor + 7) item, 1 indicates that sensor is present & 0 indicates that sensor is absent
    Sample:
    [16, 11, 5, 6]
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.013159862001600412, 0.09805189717260446, 0.0001, 0.001, 20, 5, 0.6]
    '''
    def get_feature_vector(self):
        """Returns the feature vector for this job"""
        # One-hot encode sensors (simplified for this implementation)
        sensor_vec = [0] * NUM_SENSORS
        for s in self.sensors:
            sensor_vec[s] = 1
            
        return sensor_vec + [
            self.vcpu_per_sensor,
            self.mem_per_sensor,
            self.vcpu_fl_server,
            self.mem_fl_server,
            self.epochs,
            self.model_size,
            self.training_freq
        ]
