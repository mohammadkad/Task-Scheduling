# 1404-08-20
# Mohammad Kadkhodaei
# pykalman 0.10.2, pip install pykalman, An implementation of the Kalman Filter, Kalman Smoother, and EM algorithm in Python

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
from pykalman import KalmanFilter
from sklearn.preprocessing import MinMaxScaler

# Smart factory dimensions
FACTORY_WIDTH = 100  # meters
FACTORY_HEIGHT = 80  # meters
X_NUMBER = 3  # Number of trajectories to read and plot

def read_geolife_trajectory(user_folder, max_points=1000):
    """
    Read trajectory data from a single GeoLife user folder
    """
    plt_files = glob.glob(os.path.join(user_folder, "Trajectory", "*.plt"))
    
    all_points = []
    
    for plt_file in plt_files[:2]:  # Limit to first 2 trajectory files per user
        try:
            # Skip first 6 lines (header) in GeoLife .plt files
            df = pd.read_csv(plt_file, skiprows=6, header=None, 
                           names=['lat', 'lon', 'zero', 'alt', 'days', 'date', 'time'])
            
            # Convert to numeric, handling any errors
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
            df['alt'] = pd.to_numeric(df['alt'], errors='coerce')
            
            # Drop rows with invalid coordinates
            df = df.dropna(subset=['lat', 'lon'])
            
            # Limit number of points per file
            points = df[['lat', 'lon', 'alt']].values[:max_points//2]
            all_points.extend(points)
            
        except Exception as e:
            print(f"Error reading {plt_file}: {e}")
            continue
    
    return np.array(all_points)

def map_geolife_to_factory(geolife_trajectory, factory_width=FACTORY_WIDTH, factory_height=FACTORY_HEIGHT):
    """
    Map real GeoLife GPS coordinates to factory coordinates
    """
    if len(geolife_trajectory) == 0:
        return np.array([])
    
    # Extract latitude and longitude
    lats = geolife_trajectory[:, 0]
    lons = geolife_trajectory[:, 1]
    alts = geolife_trajectory[:, 2] if geolife_trajectory.shape[1] > 2 else np.zeros(len(geolife_trajectory))
    
    # Normalize coordinates to [0, 1] range
    lat_scaler = MinMaxScaler()
    lon_scaler = MinMaxScaler()
    
    lats_normalized = lat_scaler.fit_transform(lats.reshape(-1, 1)).flatten()
    lons_normalized = lon_scaler.fit_transform(lons.reshape(-1, 1)).flatten()
    
    # Map to factory coordinates
    # Use longitude for X (factory width) and latitude for Y (factory height)
    factory_x = lons_normalized * factory_width
    factory_y = (1 - lats_normalized) * factory_height  # Invert Y to match typical coordinate systems
    
    # Map altitude to factory height (normalize to reasonable factory heights)
    alt_scaler = MinMaxScaler(feature_range=(0.5, 3.0))  # AGV heights between 0.5m and 3m
    factory_z = alt_scaler.fit_transform(alts.reshape(-1, 1)).flatten()
    
    factory_trajectory = np.column_stack([factory_x, factory_y, factory_z])
    
    return factory_trajectory

def get_geolife_based_factory_trajectories(geolife_path, num_trajectories=X_NUMBER, max_points_per_trajectory=100):
    """
    Get factory trajectories based on real GeoLife data patterns
    """
    if not os.path.exists(geolife_path):
        print(f"GeoLife path {geolife_path} not found, using simulated patterns")
        return generate_smart_factory_trajectories(num_trajectories, max_points_per_trajectory)
    
    user_folders = [f for f in os.listdir(geolife_path) 
                   if os.path.isdir(os.path.join(geolife_path, f)) and f.isdigit()]
    
    selected_users = user_folders[:num_trajectories]
    
    factory_trajectories = []
    
    for i, user_id in enumerate(selected_users):
        user_folder = os.path.join(geolife_path, user_id)
        
        if not os.path.exists(user_folder):
            continue
            
        print(f"Processing GeoLife user {user_id} for factory mapping...")
        
        # Read original GeoLife trajectory
        geolife_trajectory = read_geolife_trajectory(user_folder, max_points_per_trajectory)
        
        if len(geolife_trajectory) < 10:
            print(f"Not enough data for user {user_id}, using simulated trajectory")
            factory_trajectory = generate_single_factory_trajectory(max_points_per_trajectory)
        else:
            # Map to factory coordinates
            factory_trajectory = map_geolife_to_factory(geolife_trajectory)
            
            # If mapping fails, use simulated trajectory
            if len(factory_trajectory) == 0:
                factory_trajectory = generate_single_factory_trajectory(max_points_per_trajectory)
        
        factory_trajectories.append(factory_trajectory)
    
    # If we don't have enough GeoLife users, fill with simulated trajectories
    while len(factory_trajectories) < num_trajectories:
        factory_trajectories.append(generate_single_factory_trajectory(max_points_per_trajectory))
    
    return factory_trajectories[:num_trajectories]

def generate_single_factory_trajectory(points_per_trajectory=100):
    """
    Generate a single simulated factory trajectory
    """
    # Start from different entry points around the factory
    start_x = np.random.uniform(5, FACTORY_WIDTH-5)
    start_y = np.random.uniform(5, FACTORY_HEIGHT-5)
    
    # Simulate different movement patterns in the factory
    movement_pattern = np.random.choice(['linear', 'circular', 'random_walk', 'zigzag'])
    
    x_coords = [start_x]
    y_coords = [start_y]
    
    if movement_pattern == 'linear':
        angle = np.random.uniform(0, 2*np.pi)
        speed_x = np.cos(angle) * 0.5
        speed_y = np.sin(angle) * 0.5
        
        for i in range(1, points_per_trajectory):
            new_x = x_coords[-1] + speed_x + np.random.normal(0, 0.1)
            new_y = y_coords[-1] + speed_y + np.random.normal(0, 0.1)
            
            new_x = np.clip(new_x, 0, FACTORY_WIDTH)
            new_y = np.clip(new_y, 0, FACTORY_HEIGHT)
            
            x_coords.append(new_x)
            y_coords.append(new_y)
            
    elif movement_pattern == 'circular':
        center_x = FACTORY_WIDTH / 2
        center_y = FACTORY_HEIGHT / 2
        radius = min(FACTORY_WIDTH, FACTORY_HEIGHT) / 3
        
        for i in range(1, points_per_trajectory):
            angle = 2 * np.pi * i / points_per_trajectory
            new_x = center_x + radius * np.cos(angle) + np.random.normal(0, 0.2)
            new_y = center_y + radius * np.sin(angle) + np.random.normal(0, 0.2)
            
            x_coords.append(new_x)
            y_coords.append(new_y)
            
    elif movement_pattern == 'random_walk':
        for i in range(1, points_per_trajectory):
            new_x = x_coords[-1] + np.random.normal(0, 0.3)
            new_y = y_coords[-1] + np.random.normal(0, 0.3)
            
            new_x = np.clip(new_x, 0, FACTORY_WIDTH)
            new_y = np.clip(new_y, 0, FACTORY_HEIGHT)
            
            x_coords.append(new_x)
            y_coords.append(new_y)
            
    elif movement_pattern == 'zigzag':
        x_direction = 1
        y_direction = 1
        
        for i in range(1, points_per_trajectory):
            if i % 20 == 0:
                x_direction *= -1
            if i % 15 == 0:
                y_direction *= -1
                
            new_x = x_coords[-1] + x_direction * 0.4 + np.random.normal(0, 0.1)
            new_y = y_coords[-1] + y_direction * 0.3 + np.random.normal(0, 0.1)
            
            if new_x <= 0 or new_x >= FACTORY_WIDTH:
                x_direction *= -1
                new_x = np.clip(new_x, 0, FACTORY_WIDTH)
            if new_y <= 0 or new_y >= FACTORY_HEIGHT:
                y_direction *= -1
                new_y = np.clip(new_y, 0, FACTORY_HEIGHT)
            
            x_coords.append(new_x)
            y_coords.append(new_y)
    
    # Add height variation
    z_coords = np.ones(points_per_trajectory) * 1.5
    
    # Add workstation height variations
    workstation_heights = {'assembly': 2.0, 'inspection': 1.0, 'packaging': 1.8, 'storage': 0.5}
    
    for i in range(points_per_trajectory):
        if i % 25 == 0:
            station = np.random.choice(list(workstation_heights.keys()))
            z_coords[i:i+5] = workstation_heights[station]
    
    trajectory = np.column_stack([x_coords, y_coords, z_coords])
    return trajectory

def generate_smart_factory_trajectories(num_trajectories=X_NUMBER, points_per_trajectory=100):
    """
    Generate simulated trajectory data for a smart factory environment
    """
    trajectories = []
    for _ in range(num_trajectories):
        trajectories.append(generate_single_factory_trajectory(points_per_trajectory))
    return trajectories

def create_kalman_filter(dim=3):
    """
    Create a Kalman filter for trajectory prediction
    """
    # State transition matrix (assuming constant velocity model)
    F = np.eye(dim * 2)  # Position and velocity for each dimension
    for i in range(dim):
        F[i, dim + i] = 1  # Position + velocity
    
    # Observation matrix (we only observe position)
    H = np.eye(dim, dim * 2)
    
    # Process covariance (uncertainty in the model)
    Q = np.eye(dim * 2) * 0.1
    
    # Observation covariance (measurement noise)
    R = np.eye(dim) * 0.1
    
    kf = KalmanFilter(
        transition_matrices=F,
        observation_matrices=H,
        transition_covariance=Q,
        observation_covariance=R,
        initial_state_mean=np.zeros(dim * 2),
        initial_state_covariance=np.eye(dim * 2)
    )
    
    return kf

def predict_trajectory_kalman(trajectory, prediction_steps=10):
    """
    Use Kalman filter to predict future trajectory points
    """
    if len(trajectory) < 2:
        return trajectory, trajectory
    
    dim = trajectory.shape[1]
    kf = create_kalman_filter(dim)
    
    # Prepare measurements (positions only)
    measurements = trajectory
    
    # Initialize state with first two points to estimate velocity
    if len(trajectory) >= 2:
        initial_state = np.zeros(dim * 2)
        initial_state[:dim] = trajectory[0]
        initial_state[dim:] = trajectory[1] - trajectory[0]  # Initial velocity
        
        kf.initial_state_mean = initial_state
    
    # Filter and smooth the existing trajectory
    try:
        state_means, state_covariances = kf.filter(measurements)
        smoothed_means, smoothed_covariances = kf.smooth(measurements)
    except:
        # If filtering fails, return original trajectory
        return trajectory, trajectory
    
    # Predict future points
    predicted_means = []
    predicted_covariances = []
    
    current_mean = state_means[-1]
    current_covariance = state_covariances[-1]
    
    for _ in range(prediction_steps):
        # Predict next state
        current_mean = kf.transition_matrices.dot(current_mean)
        current_covariance = (kf.transition_matrices.dot(current_covariance).dot(
                            kf.transition_matrices.T) + kf.transition_covariance)
        
        # Extract position from state (first dim elements)
        predicted_position = kf.observation_matrices.dot(current_mean)
        predicted_means.append(predicted_position)
        predicted_covariances.append(current_covariance)
    
    predicted_trajectory = np.array(predicted_means)
    
    return smoothed_means[:, :dim], predicted_trajectory

def plot_3d_smart_factory_trajectories(geolife_path=None, prediction_steps=20, use_geolife=True):
    """
    Plot 3D trajectories with Kalman filter predictions in smart factory
    """
    if use_geolife and geolife_path:
        trajectories = get_geolife_based_factory_trajectories(geolife_path, X_NUMBER)
        data_source = "GeoLife-based"
    else:
        trajectories = generate_smart_factory_trajectories(X_NUMBER)
        data_source = "Simulated"
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, trajectory in enumerate(trajectories):
        if len(trajectory) < 10:
            print(f"Not enough trajectory data for AGV {i+1}")
            continue
        
        # Apply Kalman filter
        smoothed_trajectory, predicted_trajectory = predict_trajectory_kalman(
            trajectory, prediction_steps)
        
        # Extract coordinates
        real_x = trajectory[:, 0]
        real_y = trajectory[:, 1]
        real_z = trajectory[:, 2]
        
        smoothed_x = smoothed_trajectory[:, 0]
        smoothed_y = smoothed_trajectory[:, 1]
        smoothed_z = smoothed_trajectory[:, 2]
        
        if len(predicted_trajectory) > 0:
            predicted_x = predicted_trajectory[:, 0]
            predicted_y = predicted_trajectory[:, 1]
            predicted_z = predicted_trajectory[:, 2]
        
        color = colors[i % len(colors)]
        
        # Plot real trajectory (thin, transparent)
        ax.plot(real_x, real_y, real_z, 
               color=color, linewidth=1, alpha=0.3, 
               label=f'AGV {i+1} - Real')
        
        # Plot smoothed trajectory
        ax.plot(smoothed_x, smoothed_y, smoothed_z, 
               color=color, linewidth=2, alpha=0.7, 
               label=f'AGV {i+1} - Smoothed')
        
        # Plot predicted trajectory
        if len(predicted_trajectory) > 0:
            ax.plot(predicted_x, predicted_y, predicted_z, 
                   color=color, linewidth=2, alpha=0.7, linestyle='--',
                   label=f'AGV {i+1} - Predicted')
        
        # Plot key points
        ax.scatter(real_x[0], real_y[0], real_z[0], 
                  color=color, s=80, marker='o', edgecolors='black')
        ax.scatter(real_x[-1], real_y[-1], real_z[-1], 
                  color=color, s=80, marker='s', edgecolors='black')
        
        if len(predicted_trajectory) > 0:
            ax.scatter(predicted_x[-1], predicted_y[-1], predicted_z[-1], 
                      color=color, s=80, marker='*', edgecolors='black')
    
    # Set labels and title
    ax.set_xlabel('X Position (meters)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y Position (meters)', fontsize=12, labelpad=10)
    ax.set_zlabel('Height (meters)', fontsize=12, labelpad=10)
    ax.set_title(f'Smart Factory 3D Trajectories with Kalman Filter Prediction\nFactory: {FACTORY_WIDTH}m × {FACTORY_HEIGHT}m | {X_NUMBER} AGVs | Data: {data_source}', 
                fontsize=14, pad=20)
    
    # Set axis limits to factory dimensions
    ax.set_xlim(0, FACTORY_WIDTH)
    ax.set_ylim(0, FACTORY_HEIGHT)
    ax.set_zlim(0, 3)
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
    
    # Adjust viewing angle
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def plot_2d_smart_factory_trajectories(geolife_path=None, prediction_steps=20, use_geolife=True):
    """
    Plot 2D trajectories with Kalman filter predictions in smart factory
    """
    if use_geolife and geolife_path:
        trajectories = get_geolife_based_factory_trajectories(geolife_path, X_NUMBER)
        data_source = "GeoLife-based"
    else:
        trajectories = generate_smart_factory_trajectories(X_NUMBER)
        data_source = "Simulated"
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, trajectory in enumerate(trajectories):
        if len(trajectory) < 10:
            continue
        
        # Apply Kalman filter
        smoothed_trajectory, predicted_trajectory = predict_trajectory_kalman(
            trajectory, prediction_steps)
        
        real_x = trajectory[:, 0]
        real_y = trajectory[:, 1]
        smoothed_x = smoothed_trajectory[:, 0]
        smoothed_y = smoothed_trajectory[:, 1]
        
        color = colors[i % len(colors)]
        
        # Plot trajectories
        ax.plot(real_x, real_y, color=color, linewidth=1, alpha=0.3, 
                label=f'AGV {i+1} - Real')
        ax.plot(smoothed_x, smoothed_y, color=color, linewidth=2, alpha=0.7,
                label=f'AGV {i+1} - Smoothed')
        
        if len(predicted_trajectory) > 0:
            predicted_x = predicted_trajectory[:, 0]
            predicted_y = predicted_trajectory[:, 1]
            ax.plot(predicted_x, predicted_y, color=color, linewidth=2, 
                    alpha=0.7, linestyle='--', label=f'AGV {i+1} - Predicted')
        
        # Plot key points
        ax.scatter(real_x[0], real_y[0], color=color, s=80, marker='o', 
                  edgecolors='black')
        ax.scatter(real_x[-1], real_y[-1], color=color, s=80, marker='s', 
                  edgecolors='black')
        
        if len(predicted_trajectory) > 0:
            ax.scatter(predicted_x[-1], predicted_y[-1], color=color, s=80, 
                      marker='*', edgecolors='black')
    
    ax.set_xlabel('X Position (meters)', fontsize=12)
    ax.set_ylabel('Y Position (meters)', fontsize=12)
    ax.set_title(f'Smart Factory 2D Trajectories with Kalman Filter Prediction\nFactory: {FACTORY_WIDTH}m × {FACTORY_HEIGHT}m | {X_NUMBER} AGVs | Data: {data_source}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set axis limits to factory dimensions
    ax.set_xlim(0, FACTORY_WIDTH)
    ax.set_ylim(0, FACTORY_HEIGHT)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    geolife_dataset_path = "data/Geolife/"
    
    # Try to use real GeoLife data mapped to factory, fall back to simulated data
    plot_3d_smart_factory_trajectories(geolife_path=geolife_dataset_path, 
                                     prediction_steps=15, 
                                     use_geolife=True)
    
    plot_2d_smart_factory_trajectories(geolife_path=geolife_dataset_path, 
                                     prediction_steps=15, 
                                     use_geolife=True)