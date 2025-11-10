# 1404-08-19
# Mohammad Kadkhodaei

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob

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

def plot_3d_trajectories(geolife_path, user_ids=None, max_users=5, max_points_per_user=1000):
    """
    Plot 3D trajectories for multiple users from GeoLife dataset
    """
    if user_ids is None:
        # Get available user folders
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
        
        if len(trajectory) == 0:
            print(f"No valid trajectory data for user {user_id}")
            continue
        
        # Extract coordinates
        lats = trajectory[:, 0]
        lons = trajectory[:, 1]
        alts = trajectory[:, 2]
        
        # Plot 3D trajectory
        color = colors[i % len(colors)]
        ax.plot(lons, lats, alts, 
               color=color, linewidth=2, alpha=0.7, 
               label=f'User {user_id}')
        
        # Plot start and end points
        ax.scatter(lons[0], lats[0], alts[0], 
                  color=color, s=100, marker='o', edgecolors='black')
        ax.scatter(lons[-1], lats[-1], alts[-1], 
                  color=color, s=100, marker='s', edgecolors='black')
    
    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=12, labelpad=10)
    ax.set_ylabel('Latitude', fontsize=12, labelpad=10)
    ax.set_zlabel('Altitude (meters)', fontsize=12, labelpad=10)
    ax.set_title('3D Movement Trajectories of Different Users (GeoLife Dataset)', 
                fontsize=14, pad=20)
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
    
    # Adjust viewing angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Set equal aspect ratio
    max_range = max(ax.get_xlim()[1]-ax.get_xlim()[0], 
                   ax.get_ylim()[1]-ax.get_ylim()[0],
                   ax.get_zlim()[1]-ax.get_zlim()[0])
    
    mid_x = np.mean(ax.get_xlim())
    mid_y = np.mean(ax.get_ylim())
    mid_z = np.mean(ax.get_zlim())
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def plot_2d_trajectories(geolife_path, user_ids=None, max_users=5, max_points_per_user=1000):
    """
    Plot 2D trajectories (latitude vs longitude) for multiple users
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
        
        if len(trajectory) == 0:
            continue
        
        lats = trajectory[:, 0]
        lons = trajectory[:, 1]
        
        color = colors[i % len(colors)]
        ax.plot(lons, lats, color=color, linewidth=2, alpha=0.7, label=f'User {user_id}')
        ax.scatter(lons[0], lats[0], color=color, s=100, marker='o', edgecolors='black')
        ax.scatter(lons[-1], lats[-1], color=color, s=100, marker='s', edgecolors='black')
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('2D Movement Trajectories of Different Users (GeoLife Dataset)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Path to your GeoLife dataset
    # Download from: https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/
    # C:\Users\MK\data\Geolife\000\Trajectory
    # geolife_dataset_path = "path/to/your/Geolife_Trajectories_1.3/Data"
    geolife_dataset_path = "data/Geolife/"
    
    # If you don't have the dataset, you can create sample data for demonstration
    def create_sample_trajectories():
        """Create sample trajectory data for demonstration"""
        np.random.seed(42)
        
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['red', 'blue', 'green']
        user_names = ['User 001', 'User 002', 'User 003']
        
        for i in range(3):
            # Generate random walk trajectory
            n_points = 200
            lat_start = 39.9 + np.random.uniform(-0.1, 0.1)
            lon_start = 116.3 + np.random.uniform(-0.1, 0.1)
            alt_start = 50 + np.random.uniform(-20, 20)
            
            lat_walk = np.cumsum(np.random.normal(0, 0.001, n_points))
            lon_walk = np.cumsum(np.random.normal(0, 0.001, n_points))
            alt_walk = np.cumsum(np.random.normal(0, 0.5, n_points))
            
            lats = lat_start + lat_walk
            lons = lon_start + lon_walk
            alts = alt_start + alt_walk
            
            color = colors[i]
            ax.plot(lons, lats, alts, color=color, linewidth=2, alpha=0.7, label=user_names[i])
            ax.scatter(lons[0], lats[0], alts[0], color=color, s=100, marker='o', edgecolors='black')
            ax.scatter(lons[-1], lats[-1], alts[-1], color=color, s=100, marker='s', edgecolors='black')
        
        ax.set_xlabel('Longitude', fontsize=12, labelpad=10)
        ax.set_ylabel('Latitude', fontsize=12, labelpad=10)
        ax.set_zlabel('Altitude (meters)', fontsize=12, labelpad=10)
        ax.set_title('3D Movement Trajectories of Different Users (Sample Data)', fontsize=14, pad=20)
        ax.legend()
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        plt.show()
    
    # Try to plot real GeoLife data, fall back to sample data if not available
    try:
        if os.path.exists(geolife_dataset_path):
            # Plot 3D trajectories
            fig_3d, ax_3d = plot_3d_trajectories(geolife_dataset_path, max_users=3)
            
            # Plot 2D trajectories
            plot_2d_trajectories(geolife_dataset_path, max_users=3)
        else:
            print("GeoLife dataset not found at specified path. Creating sample trajectories...")
            create_sample_trajectories()
    except Exception as e:
        print(f"Error processing GeoLife data: {e}")
        print("Creating sample trajectories instead...")
        create_sample_trajectories()
