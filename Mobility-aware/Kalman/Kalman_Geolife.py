# 1404-08-19
# Mohammad Kadkhodaei
# pykalman 0.10.2, pip install pykalman, An implementation of the Kalman Filter, Kalman Smoother, and EM algorithm in Python

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
from pykalman import KalmanFilter

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

def plot_3d_trajectories_with_predictions(geolife_path, user_ids=None, max_users=5, max_points_per_user=1000, prediction_steps=20):
    """
    Plot 3D trajectories with Kalman filter predictions
    """
    if user_ids is None:
        user_folders = [f for f in os.listdir(geolife_path) 
                       if os.path.isdir(os.path.join(geolife_path, f)) and f.isdigit()]
        user_ids = user_folders[:max_users]
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, user_id in enumerate(user_ids):
        user_folder = os.path.join(geolife_path, user_id)
        
        if not os.path.exists(user_folder):
            print(f"User folder {user_folder} not found, skipping...")
            continue
            
        print(f"Processing user {user_id}...")
        
        # Read trajectory data
        trajectory = read_geolife_trajectory(user_folder, max_points_per_user)
        
        if len(trajectory) < 10:  # Need enough points for prediction
            print(f"Not enough trajectory data for user {user_id}")
            continue
        
        # Apply Kalman filter
        smoothed_trajectory, predicted_trajectory = predict_trajectory_kalman(
            trajectory, prediction_steps)
        
        # Extract coordinates
        real_lats = trajectory[:, 0]
        real_lons = trajectory[:, 1]
        real_alts = trajectory[:, 2]
        
        smoothed_lats = smoothed_trajectory[:, 0]
        smoothed_lons = smoothed_trajectory[:, 1]
        smoothed_alts = smoothed_trajectory[:, 2]
        
        if len(predicted_trajectory) > 0:
            predicted_lats = predicted_trajectory[:, 0]
            predicted_lons = predicted_trajectory[:, 1]
            predicted_alts = predicted_trajectory[:, 2]
        
        color = colors[i % len(colors)]
        
        # Plot real trajectory (thin, transparent)
        ax.plot(real_lons, real_lats, real_alts, 
               color=color, linewidth=1, alpha=0.3, 
               label=f'User {user_id} - Real')
        
        # Plot smoothed trajectory
        ax.plot(smoothed_lons, smoothed_lats, smoothed_alts, 
               color=color, linewidth=2, alpha=0.7, 
               label=f'User {user_id} - Smoothed')
        
        # Plot predicted trajectory
        if len(predicted_trajectory) > 0:
            ax.plot(predicted_lons, predicted_lats, predicted_alts, 
                   color=color, linewidth=2, alpha=0.7, linestyle='--',
                   label=f'User {user_id} - Predicted')
        
        # Plot key points
        ax.scatter(real_lons[0], real_lats[0], real_alts[0], 
                  color=color, s=80, marker='o', edgecolors='black', label='Start')
        ax.scatter(real_lons[-1], real_lats[-1], real_alts[-1], 
                  color=color, s=80, marker='s', edgecolors='black', label='End')
        
        if len(predicted_trajectory) > 0:
            ax.scatter(predicted_lons[-1], predicted_lats[-1], predicted_alts[-1], 
                      color=color, s=80, marker='*', edgecolors='black', label='Predicted End')
    
    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=12, labelpad=10)
    ax.set_ylabel('Latitude', fontsize=12, labelpad=10)
    ax.set_zlabel('Altitude (meters)', fontsize=12, labelpad=10)
    ax.set_title(f'3D Trajectories with Kalman Filter Prediction ({prediction_steps} steps)', 
                fontsize=14, pad=20)
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
    
    # Adjust viewing angle
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def plot_2d_trajectories_with_predictions(geolife_path, user_ids=None, max_users=5, max_points_per_user=1000, prediction_steps=20):
    """
    Plot 2D trajectories with Kalman filter predictions
    """
    if user_ids is None:
        user_folders = [f for f in os.listdir(geolife_path) 
                       if os.path.isdir(os.path.join(geolife_path, f)) and f.isdigit()]
        user_ids = user_folders[:max_users]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, user_id in enumerate(user_ids):
        user_folder = os.path.join(geolife_path, user_id)
        
        if not os.path.exists(user_folder):
            continue
            
        trajectory = read_geolife_trajectory(user_folder, max_points_per_user)
        
        if len(trajectory) < 10:
            continue
        
        # Apply Kalman filter
        smoothed_trajectory, predicted_trajectory = predict_trajectory_kalman(
            trajectory, prediction_steps)
        
        real_lats = trajectory[:, 0]
        real_lons = trajectory[:, 1]
        smoothed_lats = smoothed_trajectory[:, 0]
        smoothed_lons = smoothed_trajectory[:, 1]
        
        color = colors[i % len(colors)]
        
        # Plot trajectories
        ax.plot(real_lons, real_lats, color=color, linewidth=1, alpha=0.3, 
                label=f'User {user_id} - Real')
        ax.plot(smoothed_lons, smoothed_lats, color=color, linewidth=2, alpha=0.7,
                label=f'User {user_id} - Smoothed')
        
        if len(predicted_trajectory) > 0:
            predicted_lats = predicted_trajectory[:, 0]
            predicted_lons = predicted_trajectory[:, 1]
            ax.plot(predicted_lons, predicted_lats, color=color, linewidth=2, 
                    alpha=0.7, linestyle='--', label=f'User {user_id} - Predicted')
        
        # Plot key points
        ax.scatter(real_lons[0], real_lats[0], color=color, s=80, marker='o', 
                  edgecolors='black')
        ax.scatter(real_lons[-1], real_lats[-1], color=color, s=80, marker='s', 
                  edgecolors='black')
        
        if len(predicted_trajectory) > 0:
            ax.scatter(predicted_lons[-1], predicted_lats[-1], color=color, s=80, 
                      marker='*', edgecolors='black')
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'2D Trajectories with Kalman Filter Prediction ({prediction_steps} steps)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Install required package: pip install pykalman
    geolife_dataset_path = "data/Geolife/"
    
    def create_sample_trajectories_with_predictions():
        """Create sample trajectory data with Kalman filter predictions"""
        np.random.seed(42)
        
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['red', 'blue', 'green']
        user_names = ['User 001', 'User 002', 'User 003']
        
        for i in range(3):
            # Generate random walk trajectory
            n_points = 100
            lat_start = 39.9 + np.random.uniform(-0.1, 0.1)
            lon_start = 116.3 + np.random.uniform(-0.1, 0.1)
            alt_start = 50 + np.random.uniform(-20, 20)
            
            lat_walk = np.cumsum(np.random.normal(0, 0.001, n_points))
            lon_walk = np.cumsum(np.random.normal(0, 0.001, n_points))
            alt_walk = np.cumsum(np.random.normal(0, 0.5, n_points))
            
            lats = lat_start + lat_walk
            lons = lon_start + lon_walk
            alts = alt_start + alt_walk
            
            trajectory = np.column_stack([lats, lons, alts])
            
            # Apply Kalman filter
            smoothed_trajectory, predicted_trajectory = predict_trajectory_kalman(
                trajectory, prediction_steps=15)
            
            color = colors[i]
            
            # Plot trajectories
            ax.plot(lons, lats, alts, color=color, linewidth=1, alpha=0.3, 
                    label=f'{user_names[i]} - Real')
            ax.plot(smoothed_trajectory[:, 1], smoothed_trajectory[:, 0], smoothed_trajectory[:, 2], 
                   color=color, linewidth=2, alpha=0.7, label=f'{user_names[i]} - Smoothed')
            
            if len(predicted_trajectory) > 0:
                ax.plot(predicted_trajectory[:, 1], predicted_trajectory[:, 0], predicted_trajectory[:, 2], 
                       color=color, linewidth=2, alpha=0.7, linestyle='--', 
                       label=f'{user_names[i]} - Predicted')
        
        ax.set_xlabel('Longitude', fontsize=12, labelpad=10)
        ax.set_ylabel('Latitude', fontsize=12, labelpad=10)
        ax.set_zlabel('Altitude (meters)', fontsize=12, labelpad=10)
        ax.set_title('3D Trajectories with Kalman Filter Prediction (Sample Data)', fontsize=14, pad=20)
        ax.legend()
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        plt.show()
    
    # Try to plot real GeoLife data with predictions
    try:
        if os.path.exists(geolife_dataset_path):
            # Plot 3D trajectories with predictions
            plot_3d_trajectories_with_predictions(geolife_dataset_path, max_users=3, prediction_steps=15)
            
            # Plot 2D trajectories with predictions
            plot_2d_trajectories_with_predictions(geolife_dataset_path, max_users=3, prediction_steps=15)
        else:
            print("GeoLife dataset not found at specified path. Creating sample trajectories with predictions...")
            create_sample_trajectories_with_predictions()
    except Exception as e:
        print(f"Error processing GeoLife data: {e}")
        print("Creating sample trajectories with predictions instead...")
        create_sample_trajectories_with_predictions()