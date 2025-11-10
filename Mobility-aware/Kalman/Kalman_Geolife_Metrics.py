# 1404-08-19
# Mohammad Kadkhodaei
'''
Accuracy Interpretation:
90-100%: Excellent prediction (very close to actual path)
80-89%: Very good prediction
70-79%: Good prediction
60-69%: Reasonable prediction
Below 60%: Poor prediction (may need model tuning)
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
from pykalman import KalmanFilter
from sklearn.metrics import mean_squared_error

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
    Returns: smoothed_trajectory, predicted_trajectory, prediction_accuracy
    """
    if len(trajectory) < prediction_steps + 5:
        return trajectory, np.array([]), 0.0
    
    dim = trajectory.shape[1]
    kf = create_kalman_filter(dim)
    
    # Split data into training and validation
    split_idx = len(trajectory) - prediction_steps
    train_data = trajectory[:split_idx]
    test_data = trajectory[split_idx:]
    
    # Initialize state with first two points to estimate velocity
    if len(train_data) >= 2:
        initial_state = np.zeros(dim * 2)
        initial_state[:dim] = train_data[0]
        initial_state[dim:] = train_data[1] - train_data[0]  # Initial velocity
        kf.initial_state_mean = initial_state
    
    # Filter and smooth the training data
    try:
        state_means, state_covariances = kf.filter(train_data)
        smoothed_means, smoothed_covariances = kf.smooth(train_data)
    except Exception as e:
        print(f"Kalman filter error: {e}")
        return trajectory, np.array([]), 0.0
    
    # Predict future points
    predicted_means = []
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
    
    predicted_trajectory = np.array(predicted_means)
    
    # Calculate prediction accuracy
    if len(predicted_trajectory) > 0 and len(test_data) > 0:
        min_len = min(len(predicted_trajectory), len(test_data))
        
        # Calculate distance errors
        errors = []
        for i in range(min_len):
            # Calculate Euclidean distance between predicted and actual
            error = np.linalg.norm(predicted_trajectory[i] - test_data[i])
            errors.append(error)
        
        avg_error = np.mean(errors)
        
        # Calculate accuracy as percentage (lower error = higher accuracy)
        # Assuming typical GPS error of ~10 meters, scale accordingly
        max_reasonable_error = 100.0  # meters
        accuracy = max(0, 100 - (avg_error / max_reasonable_error * 100))
        accuracy = min(100, accuracy)  # Cap at 100%
    else:
        accuracy = 0.0
    
    return smoothed_means[:, :dim], predicted_trajectory, accuracy

def calculate_distance_accuracy(predicted, actual):
    """
    Calculate prediction accuracy based on distance errors
    """
    if len(predicted) == 0 or len(actual) == 0:
        return 0.0
    
    min_len = min(len(predicted), len(actual))
    errors = []
    
    for i in range(min_len):
        # 3D Euclidean distance
        error = np.sqrt(
            (predicted[i, 0] - actual[i, 0])**2 +
            (predicted[i, 1] - actual[i, 1])**2 +
            (predicted[i, 2] - actual[i, 2])**2
        )
        errors.append(error)
    
    avg_error = np.mean(errors)
    
    # Convert to accuracy percentage (lower error = higher accuracy)
    # Using logarithmic scale since errors can vary widely
    if avg_error == 0:
        return 100.0
    elif avg_error < 1:  # Very high accuracy (<1 meter error)
        return 95 + (1 - avg_error) * 5
    elif avg_error < 10:  # Good accuracy (<10 meters error)
        return 85 + (10 - avg_error) / 9 * 10
    elif avg_error < 50:  # Reasonable accuracy (<50 meters error)
        return 70 + (50 - avg_error) / 40 * 15
    elif avg_error < 100:  # Poor accuracy (<100 meters error)
        return 50 + (100 - avg_error) / 50 * 20
    else:  # Very poor accuracy
        return max(0, 50 - (avg_error - 100) / 100 * 50)

def plot_3d_trajectories_with_predictions(geolife_path, user_ids=None, max_users=5, max_points_per_user=1000, prediction_steps=20):
    """
    Plot 3D trajectories with Kalman filter predictions and show accuracy percentages
    """
    if user_ids is None:
        user_folders = [f for f in os.listdir(geolife_path) 
                       if os.path.isdir(os.path.join(geolife_path, f)) and f.isdigit()]
        user_ids = user_folders[:max_users]
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    accuracy_results = {}
    
    for i, user_id in enumerate(user_ids):
        user_folder = os.path.join(geolife_path, user_id)
        
        if not os.path.exists(user_folder):
            print(f"User folder {user_folder} not found, skipping...")
            continue
            
        print(f"Processing user {user_id}...")
        
        # Read trajectory data
        trajectory = read_geolife_trajectory(user_folder, max_points_per_user)
        
        if len(trajectory) < prediction_steps + 10:
            print(f"Not enough trajectory data for user {user_id} (need at least {prediction_steps + 10} points)")
            accuracy_results[user_id] = 0.0
            continue
        
        # Apply Kalman filter
        smoothed_trajectory, predicted_trajectory, accuracy = predict_trajectory_kalman(
            trajectory, prediction_steps)
        
        accuracy_results[user_id] = accuracy
        
        # Extract coordinates
        real_lats = trajectory[:, 0]
        real_lons = trajectory[:, 1]
        real_alts = trajectory[:, 2]
        
        smoothed_lats = smoothed_trajectory[:, 0]
        smoothed_lons = smoothed_trajectory[:, 1]
        smoothed_alts = smoothed_trajectory[:, 2]
        
        color = colors[i % len(colors)]
        
        # Plot real trajectory (thin, transparent)
        ax.plot(real_lons, real_lats, real_alts, 
               color=color, linewidth=1, alpha=0.3)
        
        # Plot smoothed trajectory
        ax.plot(smoothed_lons, smoothed_lats, smoothed_alts, 
               color=color, linewidth=2, alpha=0.7)
        
        # Plot predicted trajectory if available
        if len(predicted_trajectory) > 0:
            predicted_lats = predicted_trajectory[:, 0]
            predicted_lons = predicted_trajectory[:, 1]
            predicted_alts = predicted_trajectory[:, 2]
            
            ax.plot(predicted_lons, predicted_lats, predicted_alts, 
                   color=color, linewidth=2, alpha=0.7, linestyle='--')
        
        # Plot key points
        ax.scatter(real_lons[0], real_lats[0], real_alts[0], 
                  color=color, s=80, marker='o', edgecolors='black')
        ax.scatter(real_lons[-1], real_lats[-1], real_alts[-1], 
                  color=color, s=80, marker='s', edgecolors='black')
        
        if len(predicted_trajectory) > 0:
            ax.scatter(predicted_lons[-1], predicted_lats[-1], predicted_alts[-1], 
                      color=color, s=100, marker='*', edgecolors='black')
    
    # Create legend with accuracy information
    legend_handles = []
    legend_labels = []
    
    for i, user_id in enumerate(user_ids):
        if user_id in accuracy_results:
            color = colors[i % len(colors)]
            accuracy = accuracy_results[user_id]
            
            # Create proxy artists for legend
            from matplotlib.lines import Line2D
            handle = Line2D([0], [0], color=color, linewidth=3, 
                          label=f'User {user_id}: {accuracy:.1f}% accurate')
            legend_handles.append(handle)
    
    # Add general legend items
    real_line = Line2D([0], [0], color='black', linewidth=1, alpha=0.5, label='Real trajectory')
    smoothed_line = Line2D([0], [0], color='black', linewidth=2, label='Smoothed (Kalman)')
    predicted_line = Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Predicted')
    start_marker = Line2D([0], [0], marker='o', color='black', markersize=8, 
                         linestyle='None', label='Start point')
    end_marker = Line2D([0], [0], marker='s', color='black', markersize=8, 
                       linestyle='None', label='End point')
    pred_marker = Line2D([0], [0], marker='*', color='black', markersize=10, 
                        linestyle='None', label='Predicted end')
    
    legend_handles.extend([real_line, smoothed_line, predicted_line, 
                          start_marker, end_marker, pred_marker])
    
    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=12, labelpad=10)
    ax.set_ylabel('Latitude', fontsize=12, labelpad=10)
    ax.set_zlabel('Altitude (meters)', fontsize=12, labelpad=10)
    ax.set_title(f'3D Trajectories with Kalman Filter Prediction ({prediction_steps} steps)\nPrediction Accuracy by User', 
                fontsize=14, pad=20)
    
    # Add legend
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Adjust viewing angle
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print accuracy results
    print("\n" + "="*50)
    print("PREDICTION ACCURACY RESULTS")
    print("="*50)
    for user_id, accuracy in accuracy_results.items():
        print(f"User {user_id}: {accuracy:.2f}% prediction accuracy")
    
    avg_accuracy = np.mean(list(accuracy_results.values())) if accuracy_results else 0
    print(f"\nAverage prediction accuracy: {avg_accuracy:.2f}%")
    print("="*50)
    
    return fig, ax, accuracy_results

def plot_2d_trajectories_with_predictions(geolife_path, user_ids=None, max_users=5, max_points_per_user=1000, prediction_steps=20):
    """
    Plot 2D trajectories with Kalman filter predictions and show accuracy percentages
    """
    if user_ids is None:
        user_folders = [f for f in os.listdir(geolife_path) 
                       if os.path.isdir(os.path.join(geolife_path, f)) and f.isdigit()]
        user_ids = user_folders[:max_users]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    accuracy_results = {}
    
    for i, user_id in enumerate(user_ids):
        user_folder = os.path.join(geolife_path, user_id)
        
        if not os.path.exists(user_folder):
            continue
            
        trajectory = read_geolife_trajectory(user_folder, max_points_per_user)
        
        if len(trajectory) < prediction_steps + 10:
            accuracy_results[user_id] = 0.0
            continue
        
        # Apply Kalman filter
        smoothed_trajectory, predicted_trajectory, accuracy = predict_trajectory_kalman(
            trajectory, prediction_steps)
        
        accuracy_results[user_id] = accuracy
        
        real_lats = trajectory[:, 0]
        real_lons = trajectory[:, 1]
        smoothed_lats = smoothed_trajectory[:, 0]
        smoothed_lons = smoothed_trajectory[:, 1]
        
        color = colors[i % len(colors)]
        
        # Plot trajectories
        ax.plot(real_lons, real_lats, color=color, linewidth=1, alpha=0.3)
        ax.plot(smoothed_lons, smoothed_lats, color=color, linewidth=2, alpha=0.7)
        
        if len(predicted_trajectory) > 0:
            predicted_lats = predicted_trajectory[:, 0]
            predicted_lons = predicted_trajectory[:, 1]
            ax.plot(predicted_lons, predicted_lats, color=color, linewidth=2, 
                    alpha=0.7, linestyle='--')
        
        # Plot key points
        ax.scatter(real_lons[0], real_lats[0], color=color, s=80, marker='o', 
                  edgecolors='black')
        ax.scatter(real_lons[-1], real_lats[-1], color=color, s=80, marker='s', 
                  edgecolors='black')
        
        if len(predicted_trajectory) > 0:
            ax.scatter(predicted_lons[-1], predicted_lats[-1], color=color, s=100, 
                      marker='*', edgecolors='black')
    
    # Create legend with accuracy
    legend_handles = []
    for i, user_id in enumerate(user_ids):
        if user_id in accuracy_results:
            color = colors[i % len(colors)]
            accuracy = accuracy_results[user_id]
            handle = plt.Line2D([0], [0], color=color, linewidth=3, 
                              label=f'User {user_id}: {accuracy:.1f}%')
            legend_handles.append(handle)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'2D Trajectories with Kalman Filter Prediction ({prediction_steps} steps)\nPrediction Accuracy by User', fontsize=14)
    ax.legend(handles=legend_handles)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print accuracy results
    print("\n" + "="*50)
    print("2D PREDICTION ACCURACY RESULTS")
    print("="*50)
    for user_id, accuracy in accuracy_results.items():
        print(f"User {user_id}: {accuracy:.2f}% prediction accuracy")
    
    avg_accuracy = np.mean(list(accuracy_results.values())) if accuracy_results else 0
    print(f"\nAverage prediction accuracy: {avg_accuracy:.2f}%")
    print("="*50)
    
    return fig, ax, accuracy_results

# Example usage
if __name__ == "__main__":
    # Install required package: pip install pykalman scikit-learn
    geolife_dataset_path = "data/Geolife/"
    
    def create_sample_trajectories_with_predictions():
        """Create sample trajectory data with Kalman filter predictions and accuracy"""
        np.random.seed(42)
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['red', 'blue', 'green']
        user_names = ['User 001', 'User 002', 'User 003']
        
        accuracy_results = {}
        
        for i in range(3):
            # Generate random walk trajectory
            n_points = 150
            lat_start = 39.9 + np.random.uniform(-0.1, 0.1)
            lon_start = 116.3 + np.random.uniform(-0.1, 0.1)
            alt_start = 50 + np.random.uniform(-20, 20)
            
            lat_walk = np.cumsum(np.random.normal(0, 0.0005, n_points))
            lon_walk = np.cumsum(np.random.normal(0, 0.0005, n_points))
            alt_walk = np.cumsum(np.random.normal(0, 0.3, n_points))
            
            lats = lat_start + lat_walk
            lons = lon_start + lon_walk
            alts = alt_start + alt_walk
            
            trajectory = np.column_stack([lats, lons, alts])
            
            # Apply Kalman filter
            smoothed_trajectory, predicted_trajectory, accuracy = predict_trajectory_kalman(
                trajectory, prediction_steps=15)
            
            accuracy_results[user_names[i]] = accuracy
            
            color = colors[i]
            
            # Plot trajectories
            ax.plot(lons, lats, alts, color=color, linewidth=1, alpha=0.3)
            ax.plot(smoothed_trajectory[:, 1], smoothed_trajectory[:, 0], smoothed_trajectory[:, 2], 
                   color=color, linewidth=2, alpha=0.7)
            
            if len(predicted_trajectory) > 0:
                ax.plot(predicted_trajectory[:, 1], predicted_trajectory[:, 0], predicted_trajectory[:, 2], 
                       color=color, linewidth=2, alpha=0.7, linestyle='--')
        
        # Create legend with accuracy
        legend_handles = []
        for i, user_name in enumerate(user_names):
            color = colors[i]
            accuracy = accuracy_results[user_name]
            from matplotlib.lines import Line2D
            handle = Line2D([0], [0], color=color, linewidth=3, 
                          label=f'{user_name}: {accuracy:.1f}% accurate')
            legend_handles.append(handle)
        
        ax.set_xlabel('Longitude', fontsize=12, labelpad=10)
        ax.set_ylabel('Latitude', fontsize=12, labelpad=10)
        ax.set_zlabel('Altitude (meters)', fontsize=12, labelpad=10)
        ax.set_title('3D Trajectories with Kalman Filter Prediction (Sample Data)\nPrediction Accuracy by User', 
                    fontsize=14, pad=20)
        ax.legend(handles=legend_handles)
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print accuracy results
        print("\n" + "="*50)
        print("SAMPLE DATA PREDICTION ACCURACY RESULTS")
        print("="*50)
        for user_name, accuracy in accuracy_results.items():
            print(f"{user_name}: {accuracy:.2f}% prediction accuracy")
        
        avg_accuracy = np.mean(list(accuracy_results.values()))
        print(f"\nAverage prediction accuracy: {avg_accuracy:.2f}%")
        print("="*50)
    
    # Try to plot real GeoLife data with predictions
    try:
        if os.path.exists(geolife_dataset_path):
            print("Processing GeoLife dataset with Kalman filter predictions...")
            
            # Plot 3D trajectories with predictions and accuracy
            fig_3d, ax_3d, acc_3d = plot_3d_trajectories_with_predictions(
                geolife_dataset_path, max_users=3, prediction_steps=15)
            
            # Plot 2D trajectories with predictions and accuracy
            fig_2d, ax_2d, acc_2d = plot_2d_trajectories_with_predictions(
                geolife_dataset_path, max_users=3, prediction_steps=15)
        else:
            print("GeoLife dataset not found at specified path. Creating sample trajectories with predictions...")
            create_sample_trajectories_with_predictions()
    except Exception as e:
        print(f"Error processing GeoLife data: {e}")
        print("Creating sample trajectories with predictions instead...")
        create_sample_trajectories_with_predictions()