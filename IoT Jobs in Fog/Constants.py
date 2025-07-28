# by : Mohammad Kadkhodaei

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