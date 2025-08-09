# 1404-05-06
# by : Mohammad Kadkhodaei
# for: Deep reinforcement learning-based optimal deployment of IoT machine learning jobs in fog computing architecture
# doi: https://doi.org/10.1007/s00607-024-01353-3

# ---
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math

# Constants based on the paper's parameters
NUM_JOBS = 100
NUM_EPISODES = 120
NUM_SENSORS = 21
NUM_FOG_NODES = 7
NUM_EDGE_NODES = 14
NUM_CLOUD_NODES = 1

# Resource demands (random ranges from paper)
MIN_vCPU_PER_SENSOR = 0.01
MAX_vCPU_PER_SENSOR = 0.1
MIN_MEM_PER_SENSOR = 0.01  # GB
MAX_MEM_PER_SENSOR = 0.1   # GB
vCPU_PER_FL_SERVER = 0.0001
MEM_PER_FL_SERVER = 0.001  # GB
FL_EPOCHS = 20
MODEL_SIZE = 5  # MB
FL_TRAINING_FREQ = 0.6

# Infrastructure parameters
CLOUD_vCPU = float('inf')  # Elastic
CLOUD_MEM = float('inf')   # Elastic
FOG_vCPU = 4
FOG_MEM = 4  # GB
EDGE_vCPU = 1
EDGE_MEM = 1  # GB

# Costs (based on AWS Fargate pricing)
CLOUD_vCPU_COST = 0.97152  # $/vCPU/Day
CLOUD_MEM_COST = 0.10668   # $/GB/Day
FOG_vCPU_COST = 0.4        # $/vCPU/Day
FOG_MEM_COST = 0.05        # $/GB/Day
EDGE_vCPU_COST = 0.0
EDGE_MEM_COST = 0.0

# Data transfer costs
TYPE1_COST = 0.09  # $/GB (cloud to internet)
TYPE2_COST = 0.32  # $/GB (internet)
TYPE3_COST = 0.16  # $/GB (intranet)

# DDPG parameters
ACTOR_LR = 0.0005
CRITIC_LR = 0.002
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = 100000
BATCH_SIZE = 256
NOISE_STD = 0.1
NOISE_DECAY = 0.9995

# Task creation parameters
SEGMENT_SIZE = 2 / (NUM_SENSORS * 10)
# ---

'''
1404-05-15, Completed
'''
# --- JOB
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
# --- JOB

'''
1404-05-18, 
'''

# --- Infra
class Infrastructure:
    def __init__(self):
    
        '''
        Sample: [{'remaining_vcpu': inf, 'remaining_mem': inf}]
        '''
        self.cloud_nodes = [{'remaining_vcpu': CLOUD_vCPU, 'remaining_mem': CLOUD_MEM} for _ in range(NUM_CLOUD_NODES)]
        '''
        Sample:
        [{'remaining_vcpu': 4, 'remaining_mem': 4}, {'remaining_vcpu': 4, 'remaining_mem': 4}, {'remaining_vcpu': 4, 'remaining_mem': 4}, {'remaining_vcpu': 4, 'remaining_mem': 4}, {'remaining_vcpu': 4, 'remaining_mem': 4}, {'remaining_vcpu': 4, 'remaining_mem': 4}, {'remaining_vcpu': 4, 'remaining_mem': 4}]
        '''
        self.fog_nodes = [{'remaining_vcpu': FOG_vCPU, 'remaining_mem': FOG_MEM} for _ in range(NUM_FOG_NODES)]
        self.edge_nodes = [{'remaining_vcpu': EDGE_vCPU, 'remaining_mem': EDGE_MEM} for _ in range(NUM_EDGE_NODES)]
        

        
    '''
    Sample: 7+14+1 = 22 and 22*2(cpu,mem) = 44 
    [inf, inf, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    '''
    def get_feature_vector(self):
        """Returns the current state of infrastructure resources"""
        cloud_vcpu = [n['remaining_vcpu'] for n in self.cloud_nodes]
        cloud_mem = [n['remaining_mem'] for n in self.cloud_nodes]
        fog_vcpu = [n['remaining_vcpu'] for n in self.fog_nodes]
        fog_mem = [n['remaining_mem'] for n in self.fog_nodes]
        edge_vcpu = [n['remaining_vcpu'] for n in self.edge_nodes]
        edge_mem = [n['remaining_mem'] for n in self.edge_nodes]
        
        return cloud_vcpu + cloud_mem + fog_vcpu + fog_mem + edge_vcpu + edge_mem
# --- Infra

def main():
  # 1404-05-07
  # Initialize components
  # ...

  # DDPG agent (state dim = job features + infra features)
  # job_feature_dim = ? # TODO, need to define.
  # infra_feature_dim = ? # TODO, need to define.


  for episode in range(NUM_EPISODES):
    # Create jobs
    jobs = [Job(i) for i in range(NUM_JOBS)]
  print(jobs)      


  print("\nFinal Results:")
  
if __name__ == "__main__":
    main()






