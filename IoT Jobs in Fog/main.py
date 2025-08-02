# 1404-05-06
# by : Mohammad Kadkhodaei
# for: Deep reinforcement learning-based optimal deployment of IoT machine learning jobs in fog computing architecture
# doi: https://doi.org/10.1007/s00607-024-01353-3


# Constants based on the paper's parameters
NUM_JOBS = 100
NUM_EPISODES = 120


class Job:
  pass


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

