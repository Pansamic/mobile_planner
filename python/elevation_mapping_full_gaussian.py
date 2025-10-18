import numpy as np
import open3d as o3d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml


def load_config(config_path='config/config.yaml'):
    """
    Load configuration from YAML file.
    
    Parameters:
    config_path: Path to the config file
    
    Returns:
    config: Dictionary with configuration parameters
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def point_cloud_process(point_cloud, grid_size=None, max_height=None, max_points_in_grid=None):
    """
    Process point cloud by grid filtering.
    
    Parameters:
    point_cloud: Open3D point cloud object
    grid_size: Size of grid cells (meters)
    max_height: Maximum height threshold
    max_points_in_grid: Maximum points to keep in each grid cell
    
    Returns:
    processed_points: numpy array of processed points
    """
    # Load config if parameters are not provided
    if grid_size is None or max_height is None or max_points_in_grid is None:
        config = load_config()
        if grid_size is None:
            grid_size = config['grid_map']['resolution']
        if max_height is None:
            max_height = config['elevation_map']['max_height']
        if max_points_in_grid is None:
            max_points_in_grid = config['grid_map']['max_top_points_in_grid']
    
    points = np.asarray(point_cloud.points)
    
    # Filter points by max height (z-coordinate)
    valid_indices = points[:, 2] < max_height
    filtered_points = points[valid_indices]
    
    if len(filtered_points) == 0:
        return np.empty((0, 3))
    
    # Calculate grid indices for each point
    grid_x = np.floor(filtered_points[:, 0] / grid_size).astype(int)
    grid_y = np.floor(filtered_points[:, 1] / grid_size).astype(int)
    
    # Group points by grid cells
    unique_grids = np.unique(np.column_stack((grid_x, grid_y)), axis=0)
    
    processed_points = []
    
    for grid in unique_grids:
        # Find points in this grid cell
        mask = (grid_x == grid[0]) & (grid_y == grid[1])
        grid_points = filtered_points[mask]
        
        # Keep at most max_points_in_grid highest points in each grid
        if len(grid_points) > max_points_in_grid:
            # Sort by height (z-coordinate) and keep the highest ones
            sorted_indices = np.argsort(grid_points[:, 2])[::-1]  # Descending order
            grid_points = grid_points[sorted_indices[:max_points_in_grid]]
        
        processed_points.append(grid_points)
    
    if processed_points:
        return np.vstack(processed_points)
    else:
        return np.empty((0, 3))


def build_elevation_map(point_cloud, rate_down_sample=1.0, roll=0, pitch=0, max_height=None):
    """
    Build elevation map from point cloud using Gaussian Process regression.
    
    Parameters:
    point_cloud: Open3D point cloud object
    rate_down_sample: Downsampling rate (0.0 - 1.0)
    roll: Roll angle correction (radians)
    pitch: Pitch angle correction (radians)
    max_height: Maximum height threshold
    
    Returns:
    Mh: Elevation map
    Msigma: Uncertainty map
    xq: X coordinates of grid
    yq: Y coordinates of grid
    """
    
    # Load config if max_height is not provided
    if max_height is None:
        config = load_config()
        max_height = config['elevation_map']['max_height']
    
    # Downsample point cloud for speed if needed
    if rate_down_sample < 1.0:
        point_cloud_down = point_cloud.random_down_sample(rate_down_sample)
        points = np.asarray(point_cloud_down.points)
    else:
        points = np.asarray(point_cloud.points)
    
    # Robot orientation correction: Apply roll and pitch correction
    # Rotation matrices around X (roll) and Y (pitch)
    rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    # Combined rotation: first pitch, then roll
    r = np.dot(rx, ry)
    
    # Rotate points to align ground plane with XY-world
    xyz_rotated = np.dot(r, points.T).T
    xr = xyz_rotated[:, 0]
    yr = xyz_rotated[:, 1]
    zr = xyz_rotated[:, 2]
    
    # Filter points by max height
    valid_indices = zr < max_height
    px = xr[valid_indices]
    py = yr[valid_indices]
    pz = zr[valid_indices]
    
    # Define local grid map parameters
    map_x_min = np.min(px)
    map_x_max = np.max(px)
    map_y_min = np.min(py)
    map_y_max = np.max(py)
    
    # Load grid resolution from config
    config = load_config()
    resolution = config['grid_map']['resolution']
    
    # Create meshgrid
    x_range = np.arange(map_x_min, map_x_max - resolution, resolution)
    y_range = np.arange(map_y_min, map_y_max - resolution, resolution)
    xq, yq = np.meshgrid(x_range, y_range)
    query_points = np.column_stack((xq.ravel(), yq.ravel()))
    
    # Prepare training data
    x_train = np.column_stack((px, py))  # Inputs: 2D locations
    y_train = pz  # Output: elevations
    
    # Get Gaussian Process parameters from config
    gp_config = config['elevation_map']['gaussian_process']
    l = gp_config['l']
    sigma_f = gp_config['sigma_f']
    sigma_n = gp_config['sigma_n']
    
    # Fit GP with RBF kernel - optimized settings for speed
    # Using a simpler kernel and fewer optimization restarts for performance
    kernel = C(sigma_f, constant_value_bounds="fixed") * RBF([l, l], length_scale_bounds="fixed") + \
             WhiteKernel(sigma_n, noise_level_bounds="fixed")
    
    # Reduced complexity for faster computation
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=5, random_state=42)
    gpr.fit(x_train, y_train)
    
    # Predict elevation and variance over grid
    elevation_pred, pred_std = gpr.predict(query_points, return_std=True)
    
    # Reshape into grid maps
    mh = elevation_pred.reshape(xq.shape)  # Elevation map
    msigma = pred_std.reshape(xq.shape)    # Uncertainty (std dev) map
    
    return mh, msigma, xq, yq


def visualize_elevation_map(mh, msigma, xq, yq, original_point_cloud, processed_points):
    """
    Create a single figure with 2x2 plots: full original point cloud, processed point cloud, 
    3D elevation map, 3D uncertainty map.
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Get original point cloud data
    original_points = np.asarray(original_point_cloud.points)
    orig_px = original_points[:, 0]
    orig_py = original_points[:, 1]
    orig_pz = original_points[:, 2]
    
    # Subplot 1: Full original point cloud (3D)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    scatter1 = ax1.scatter(orig_px, orig_py, orig_pz, c=orig_pz, s=1, cmap='viridis')
    ax1.set_title('Full Original Point Cloud')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    fig.colorbar(scatter1, ax=ax1)
    
    # Subplot 2: Processed point cloud (3D)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    if processed_points.size > 0:
        scatter2 = ax2.scatter(processed_points[:, 0], processed_points[:, 1], 
                              processed_points[:, 2], c=processed_points[:, 2], s=1, cmap='viridis')
        fig.colorbar(scatter2, ax=ax2)
    else:
        ax2.text(0.5, 0.5, 0.5, 'No Data', transform=ax2.transAxes, ha='center', va='center')
    ax2.set_title('Processed Point Cloud')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    
    # Subplot 3: 3D elevation map
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    surf3 = ax3.plot_surface(xq, yq, mh, cmap='hot', edgecolor='none')
    ax3.set_title('3D Elevation Map')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Elevation (m)')
    fig.colorbar(surf3, ax=ax3, shrink=0.5)
    
    # Subplot 4: 3D uncertainty map
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    surf4 = ax4.plot_surface(xq, yq, msigma, cmap='jet', edgecolor='none')
    ax4.set_title('3D Uncertainty Map')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Uncertainty (m)')
    fig.colorbar(surf4, ax=ax4, shrink=0.5)
    
    plt.tight_layout()
    plt.show()


def main():
    # Load configuration
    config = load_config()
    
    # Extract parameters from config
    grid_resolution = config['grid_map']['resolution']
    max_height = config['elevation_map']['max_height']
    max_top_points_in_grid = config['grid_map']['max_top_points_in_grid']
    
    # Load point cloud
    filename = 'assets/room.ply'  # Replace with your actual file
    original_point_cloud = o3d.io.read_point_cloud(filename)
    
    # Process point cloud
    processed_points = point_cloud_process(
        original_point_cloud, 
        grid_size=grid_resolution, 
        max_height=max_height, 
        max_points_in_grid=max_top_points_in_grid
    )
    
    # Create a new point cloud from processed points for elevation mapping
    processed_point_cloud = o3d.geometry.PointCloud()
    processed_point_cloud.points = o3d.utility.Vector3dVector(processed_points)
    
    # Build elevation map
    mh, msigma, xq, yq = build_elevation_map(processed_point_cloud, 1.0, 0, 0, max_height)
    
    # Create visualization
    visualize_elevation_map(mh, msigma, xq, yq, original_point_cloud, processed_points)
    
    print('Elevation and uncertainty maps generated.')


if __name__ == "__main__":
    main()