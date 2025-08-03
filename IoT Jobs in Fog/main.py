# 1404-05-06
# by : Mohammad Kadkhodaei
# for: Deep reinforcement learning-based optimal deployment of IoT machine learning jobs in fog computing architecture
# doi: https://doi.org/10.1007/s00607-024-01353-3

import random

# Constants based on the paper's parameters
NUM_JOBS = 100
NUM_EPISODES = 120

# Resource demands (random ranges from paper)
MIN_vCPU_PER_SENSOR = 0.01
MAX_vCPU_PER_SENSOR = 0.1
MIN_MEM_PER_SENSOR = 0.01  # GB
MAX_MEM_PER_SENSOR = 0.1   # GB


class Job:
    def __init__(self, job_id):
        self.job_id = job_id
        '''
        1404-05-11
        random.randint(3, NUM_SENSORS//2), random.randint(3, 10) - This generates a random integer between 3 and 10, inclusive
        Sample: 3, 4, 5, 6, 7, 8, 9, or 10.


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


def main():
  # 1404-05-07
  # Initialize components
  # ...

  # DDPG agent (state dim = job features + infra features)
  job_feature_dim = ? # TODO, need to define.
  infra_feature_dim = ? # TODO, need to define.


  for episode in range(NUM_EPISODES):
    # Create jobs
    jobs = [Job(i) for i in range(NUM_JOBS)]


  
  
  print("\nFinal Results:")
  
if __name__ == "__main__":
    main()


