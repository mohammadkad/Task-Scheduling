from edge_sim_py import User
from edge_sim_py import Application
from edge_sim_py import Service


class IoTUser(User):
    def __init__(self, workload: float = 50):
        super().__init__()
        self.workload = workload  # WL_i: Data publishing rate (MB/day)
        self.coverage_task_id = None  # ID of the ML Task this sensor is assigned to

class MLJob(Application):
    def __init__(self, epoch_no: int, model_size: float, training_frequency: float):
        super().__init__()
        self.epoch_no = epoch_no
        self.model_size = model_size  # modelSize_j
        self.training_frequency = training_frequency  # TF_j
        self.tasks = []  # List of custom MLTask objects (not just Service IDs)
        self.fl_server = None

class MLTask(Service):
    def __init__(self, sensors: list = []):
        super().__init__()
        self.sensors = sensors  # List of IoTUser objects in this task's coverage (COV)
        self.is_fl_server = False  # Flag to identify the FL Server task

# Implement the Core Algorithm in a Custom Orchestrator, drl_greedy_orchestrator.py
# ---
import numpy as np
from edge_sim_py import Orchestrator

class DRLGreedyOrchestrator(Orchestrator):
    def __init__(self, ddpg_agent):
        super().__init__()
        self.agent = ddpg_agent  # Your implemented DDPG agent (e.g., using PyTorch)
        self.jobs = []  # List of MLJob objects to be deployed

    def create_tasks_and_deploy(self, job: MLJob):
        """This function implements the TWO-PHASED method."""

        # PHASE 1: DRL-Based Task Creation
        # 1. Build State Vector (s_t)
        state = self._get_state(job)

        # 2. Get Action from DRL Agent (a_t)
        sensor_positions = self.agent.get_action(state)  # e.g., [-0.8, 0.2, 0.5, ...]

        # 3. Group Sensors into Tasks (Map action to task creation)
        task_sensor_groups = self._group_sensors(job.sensors, sensor_positions)

        # 4. Create MLTask and FL Server Objects for the job
        for group in task_sensor_groups:
            new_task = MLTask(sensors=group)
            # Set task resource demands based on number of sensors in group
            new_task.cpu = 0.05 * len(group)  # Example calculation
            new_task.memory = 0.05 * len(group)
            job.tasks.append(new_task)

        # Create the FL Server task
        fl_server = MLTask()
        fl_server.is_fl_server = True
        fl_server.cpu = 0.1 * len(job.tasks)  # Demand depends on number of tasks
        fl_server.memory = 0.01 * len(job.tasks)
        job.fl_server = fl_server

        # PHASE 2: Greedy Deployment
        # Implement Algorithm 1 from the paper here
        all_tasks_to_deploy = job.tasks + [job.fl_server]

        # Sort tasks by size (largest first) as per the paper's greedy algorithm
        all_tasks_to_deploy.sort(key=lambda task: task.cpu, reverse=True)

        for task in all_tasks_to_deploy:
            best_server = None
            best_score = -float('inf')

            # Iterate over all potential servers (edge, fog, cloud)
            for server in self.model.edge_servers + self.model.cloud_servers:
                if self._has_capacity(server, task):
                    # Calculate deployment score for this (task, server) pair
                    score = self._calc_deployment_score(task, server, job)
                    if score > best_score:
                        best_score = score
                        best_server = server

            # Deploy the task to the best server found
            if best_server is not None:
                task.server = best_server
                # Update the server's available capacity
                best_server.cpu -= task.cpu
                best_server.memory -= task.memory
            else:
                print(f"Could not deploy task {task.id}! Not enough resources.")

        # Calculate Reward (r_t)
        latency = self._calculate_job_latency(job)
        cost = self._calculate_job_cost(job)
        loss = len(job.tasks)  # Using number of tasks as proxy for loss

        reward = -1 * (w1 * latency + w2 * cost + w3 * loss) # From paper's formula (9)
        self.agent.update(state, sensor_positions, reward) # Feed experience to DRL agent

    def _get_state(self, job: MLJob) -> np.array:
        """Builds the state feature vector for the DRL agent."""
        state = []
        # Job features (from jobFeatureVector_t)
        state.append(len(job.sensors))
        state.append(job.epoch_no)
        state.append(job.model_size)
        # ... add all other job features

        # Infrastructure features (from infrastructureFeatureVector_t)
        for server in self.model.edge_servers + self.model.cloud_servers:
            state.append(server.cpu)  # Remaining CPU
            state.append(server.memory) # Remaining RAM
        return np.array(state)

    def _group_sensors(self, sensors: list, positions: list) -> list:
        """Groups sensors based on their 1D positions (Paper's Task Creation component)."""
        # Implements the logic shown in Fig. 2 of the paper
        segments = np.linspace(-1, 1, num=len(sensors)//3) # Create segments
        groups = [[] for _ in segments]
        for sensor, pos in zip(sensors, positions):
            segment_index = np.digitize(pos, segments) - 1
            groups[segment_index].append(sensor)
        # Filter out empty groups
        return [group for group in groups if group]
    # ... (Implement other helper methods _has_capacity, _calc_deployment_score, etc.)
  #---

# Build the Simulation Scenario & Run It
# ---
import edge_sim_py
from drl_greedy_orchestrator import DRLGreedyOrchestrator
from my_ddpg_agent import DDPGAgent  # Your implemented DDPG agent

def main():
    # 1. Initialize Simulator and Load Dataset (e.g., your customized dataset)
    simulator = edge_sim_py.EdgeSimulator(
        objects=[],
        load_dataset="path/to/your/custom_dataset.json" # You need to create this
    )

    # 2. Initialize your DRL Agent
    ddpg_agent = DDPGAgent(state_size=100, action_size=50) # Example sizes

    # 3. Replace the default orchestrator with your custom one
    simulator.orch = DRLGreedyOrchestrator(ddpg_agent)

    # 4. Add your custom ML Jobs to the orchestrator's queue
    # (This would likely be done by creating events in the dataset)
    for job in list_of_ml_jobs:
        simulator.orch.jobs.append(job)

    # 5. Add a hook to trigger your algorithm when a job arrives
    # This is often done by extending the simulator's `process_events` method
    # or by adding a custom event like {"type": "start_ml_job", "job": job_id, "tick": 1}

    # 6. Run the simulation for a number of steps (ticks)
    simulator.run(algorithm=simulator.orch, duration=1000)

    # 7. Collect and analyze results
    results = {
        "latency": [],
        "cost": [],
        "loss": [],
        "deployment_score": []
    }
    for job in simulator.orch.jobs:
        results["latency"].append(job.latency)
        # ... populate other metrics
    # Plot results and compare against baselines (Cloud-IoT, Edge-IoT)
# ---

