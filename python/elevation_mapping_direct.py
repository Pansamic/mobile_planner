import numpy as np
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import os


def point_cloud_process_direct(point_cloud, grid_size=0.1, max_height=1.5, max_top_points_in_grid=5):
    """
    Process point cloud by grid filtering for direct approach.
    
    Parameters:
    point_cloud: Open3D point cloud object
    grid_size: Size of grid cells (meters)
    max_height: Maximum height threshold
    max_top_points_in_grid: Maximum top points to keep in each grid cell
    
    Returns:
    processed_points: numpy array of processed points
    """
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
        
        # Keep at most max_top_points_in_grid highest points in each grid
        if len(grid_points) > max_top_points_in_grid:
            # Sort by height (z-coordinate) and keep the highest ones
            sorted_indices = np.argsort(grid_points[:, 2])[::-1]  # Descending order
            grid_points = grid_points[sorted_indices[:max_top_points_in_grid]]
        
        processed_points.append(grid_points)
    
    if processed_points:
        return np.vstack(processed_points)
    else:
        return np.empty((0, 3))


def build_elevation_map_direct(points, resolution=0.1):
    """
    Build elevation map from point cloud using direct statistical approach.
    
    Parameters:
    points: numpy array of 3D points
    resolution: grid resolution in meters
    
    Returns:
    mean_map: Elevation map with mean heights
    std_map: Uncertainty map with standard deviations
    xq: X coordinates of grid
    yq: Y coordinates of grid
    """
    if len(points) == 0:
        return None, None, None, None
        
    px = points[:, 0]
    py = points[:, 1]
    pz = points[:, 2]
    
    # Define local grid map parameters
    map_x_min = np.min(px)
    map_x_max = np.max(px)
    map_y_min = np.min(py)
    map_y_max = np.max(py)
    
    # Create meshgrid
    x_range = np.arange(map_x_min, map_x_max, resolution)
    y_range = np.arange(map_y_min, map_y_max, resolution)
    xq, yq = np.meshgrid(x_range, y_range)
    
    # Initialize maps
    mean_map = np.full(xq.shape, np.nan)
    std_map = np.full(xq.shape, np.nan)
    
    # Calculate grid indices for each point
    grid_x_indices = np.floor((px - map_x_min) / resolution).astype(int)
    grid_y_indices = np.floor((py - map_y_min) / resolution).astype(int)
    
    # Clip indices to valid range
    grid_x_indices = np.clip(grid_x_indices, 0, xq.shape[1] - 1)
    grid_y_indices = np.clip(grid_y_indices, 0, xq.shape[0] - 1)
    
    # Group points by grid cells and calculate statistics
    for i in range(xq.shape[0]):
        for j in range(xq.shape[1]):
            # Find points in this grid cell
            mask = (grid_y_indices == i) & (grid_x_indices == j)
            grid_points_z = pz[mask]
            
            # Calculate mean and std if there are points in the cell
            if len(grid_points_z) > 0:
                mean_map[i, j] = np.mean(grid_points_z)
                std_map[i, j] = np.std(grid_points_z)
    
    return mean_map, std_map, xq, yq


def fill_nan_with_minimum_neighbors(elevation_map):
    """
    Replace NaN cells with minimum elevation value of surrounding cells.
    
    This function fills NaN values in an elevation map by replacing them
    with the minimum elevation value found in their 3x3 neighborhood.
    
    Parameters:
    elevation_map: 2D numpy array with NaN values representing unknown elevations
    
    Returns:
    filled_map: 2D numpy array with NaN values replaced by minimum neighbor values
    """
    # Create a copy to avoid modifying the original map
    filled_map = elevation_map.copy()
    
    # Get map dimensions
    rows, cols = filled_map.shape
    
    # Process each cell
    for i in range(rows):
        for j in range(cols):
            # Only process NaN values
            if np.isnan(filled_map[i, j]):
                # Find minimum value in the 3x3 window
                min_value = np.inf
                found_valid = False
                
                # Check the 3x3 window centered on (i, j)
                for wi in range(-1, 2):  # -1, 0, 1
                    for wj in range(-1, 2):  # -1, 0, 1
                        ni = i + wi
                        nj = j + wj
                        
                        # Check bounds
                        if 0 <= ni < rows and 0 <= nj < cols:
                            # Check if value is not NaN
                            if not np.isnan(filled_map[ni, nj]):
                                if filled_map[ni, nj] < min_value:
                                    min_value = filled_map[ni, nj]
                                    found_valid = True
                
                # If we found a valid value, use it
                if found_valid:
                    filled_map[i, j] = min_value

    return filled_map


def boxblur_filter(elevation_map, kernel_size=3):
    """
    Apply box blur filter to the elevation map.
    
    Parameters:
    elevation_map: 2D numpy array of elevation values
    kernel_size: Size of the kernel for box blur (default: 3)
    
    Returns:
    boxblur_map: Map processed with box-blur filter
    """
    boxblur_map = ndimage.uniform_filter(elevation_map, size=kernel_size)
    return boxblur_map


def gaussian_filter_func(elevation_map, sigma=1.0):
    """
    Apply Gaussian filter to the elevation map.
    
    Parameters:
    elevation_map: 2D numpy array of elevation values
    sigma: Standard deviation for Gaussian kernel (default: 1.0)
    
    Returns:
    gaussian_map: Map processed with Gaussian filter
    """
    gaussian_map = gaussian_filter(elevation_map, sigma=sigma)
    return gaussian_map


def bilateral_filter(elevation_map, spatial_sigma=2.0, intensity_sigma=0.1, kernel_size=3):
    """
    Apply bilateral filter to the elevation map from its mathematical definition.
    
    The bilateral filter combines spatial proximity and intensity similarity:
    w(i,j) = exp(-(||i-j||^2)/(2*sigma_s^2)) * exp(-(||f(i)-f(j)||^2)/(2*sigma_r^2))
    
    Parameters:
    elevation_map: 2D numpy array of elevation values
    spatial_sigma: Standard deviation for spatial kernel (default: 1.0)
    intensity_sigma: Standard deviation for intensity kernel (default: 0.1)
    kernel_size: Size of the kernel (default: 5)
    
    Returns:
    bilateral_map: Map processed with bilateral filter
    """
    # Create output array
    bilateral_map = np.zeros_like(elevation_map)
    
    # Pad the input map to handle border pixels
    pad = kernel_size // 2
    padded_map = np.pad(elevation_map, pad, mode='edge')
    
    # Precompute spatial weights
    spatial_weights = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            distance = (i - center)**2 + (j - center)**2
            spatial_weights[i, j] = np.exp(-distance / (2 * spatial_sigma**2))
    
    # Apply bilateral filter
    rows, cols = elevation_map.shape
    for i in range(rows):
        for j in range(cols):
            if not np.isnan(elevation_map[i, j]):
                # Get local window
                window = padded_map[i:i+kernel_size, j:j+kernel_size]
                
                # Compute intensity weights
                center_value = elevation_map[i, j]
                intensity_diffs = (window - center_value)**2
                intensity_weights = np.exp(-intensity_diffs / (2 * intensity_sigma**2))
                
                # Combine spatial and intensity weights
                weights = spatial_weights * intensity_weights
                
                # Handle NaN values in the window
                mask = np.isnan(window)
                weights[mask] = 0
                
                # Normalize weights
                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    weights = weights / weight_sum
                    bilateral_map[i, j] = np.sum(weights * window)
                else:
                    bilateral_map[i, j] = elevation_map[i, j]
            else:
                bilateral_map[i, j] = np.nan
    
    return bilateral_map


def set_view_angle(ax):
    """
    Set the view angle for 3D plots to top view with 20 degree angle in Y.
    
    Parameters:
    ax: 3D axis object
    """
    ax.view_init(elev=70, azim=90)  # 20 degree elevation, azimuth for Y view


def visualize_all_maps(mean_map, std_map, xq, yq, original_point_cloud, processed_points, 
                      filled_map, bilateral_map, boxblur_map, gaussian_map):
    """
    Create separate figures for all plots:
    1. Original point cloud
    2. Processed point cloud
    3. Raw 3D elevation map
    4. Uncertainty map
    5. NaN filled elevation map
    6. Box-blur filtered elevation map
    7. Bilateral filtered elevation map
    8. Gaussian filtered elevation map
    """
    # Ensure assets directory exists
    assets_dir = 'temp/figure/maps_direct_static_py'
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    
    # Get original point cloud data
    original_points = np.asarray(original_point_cloud.points)
    orig_px = original_points[:, 0]
    orig_py = original_points[:, 1]
    orig_pz = original_points[:, 2]
    
    # Figure 1: Full original point cloud (3D)
    fig1 = plt.figure(figsize=(10, 8), dpi=300)
    ax1 = fig1.add_subplot(111, projection='3d')
    scatter1 = ax1.scatter(orig_px, orig_py, orig_pz, c=orig_pz, s=1, cmap='viridis')
    ax1.set_title('Full Original Point Cloud')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    set_view_angle(ax1)  # Set view angle
    fig1.colorbar(scatter1, ax=ax1)
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, '1_original_point_cloud.png'))
    # plt.show()
    plt.close(fig1)
    
    # Figure 2: Processed point cloud (3D)
    fig2 = plt.figure(figsize=(10, 8), dpi=300)
    ax2 = fig2.add_subplot(111, projection='3d')
    if processed_points.size > 0:
        scatter2 = ax2.scatter(processed_points[:, 0], processed_points[:, 1], 
                              processed_points[:, 2], c=processed_points[:, 2], s=1, cmap='viridis')
        fig2.colorbar(scatter2, ax=ax2)
    else:
        ax2.text(0.5, 0.5, 0.5, 'No Data', transform=ax2.transAxes, ha='center', va='center')
    ax2.set_title('Processed Point Cloud')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    set_view_angle(ax2)  # Set view angle
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, '2_processed_point_cloud.png'))
    # plt.show()
    plt.close(fig2)
    
    # Figure 3: Raw 3D elevation map
    fig3 = plt.figure(figsize=(10, 8), dpi=300)
    ax3 = fig3.add_subplot(111, projection='3d')
    if mean_map is not None:
        # Mask NaN values for plotting
        masked_mean_map = np.ma.masked_where(np.isnan(mean_map), mean_map)
        surf3 = ax3.plot_surface(xq, yq, masked_mean_map, cmap='jet', edgecolor='none')
        ax3.set_title('Raw 3D Elevation Map')
        fig3.colorbar(surf3, ax=ax3, shrink=0.5)
    else:
        ax3.text(0.5, 0.5, 0.5, 'No Data', transform=ax3.transAxes, ha='center', va='center')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Elevation (m)')
    set_view_angle(ax3)  # Set view angle
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, '3_raw_elevation_map.png'))
    # plt.show()
    plt.close(fig3)
    
    # Figure 4: Uncertainty map
    fig4 = plt.figure(figsize=(10, 8), dpi=300)
    ax4 = fig4.add_subplot(111, projection='3d')
    if std_map is not None:
        # Mask NaN values for plotting
        masked_std_map = np.ma.masked_where(np.isnan(std_map), std_map)
        surf4 = ax4.plot_surface(xq, yq, masked_std_map, cmap='jet', edgecolor='none')
        ax4.set_title('Uncertainty Map')
        fig4.colorbar(surf4, ax=ax4, shrink=0.5)
    else:
        ax4.text(0.5, 0.5, 0.5, 'No Data', transform=ax4.transAxes, ha='center', va='center')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Uncertainty (m)')
    set_view_angle(ax4)  # Set view angle
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, '4_uncertainty_map.png'))
    # plt.show()
    plt.close(fig4)
    
    # Figure 5: NaN-filled elevation map
    fig5 = plt.figure(figsize=(10, 8), dpi=300)
    ax5 = fig5.add_subplot(111, projection='3d')
    if filled_map is not None and not np.isnan(filled_map).all():
        masked_filled = np.ma.masked_where(np.isnan(filled_map), filled_map)
        surf5 = ax5.plot_surface(xq, yq, masked_filled, cmap='jet', edgecolor='none')
        ax5.set_title('NaN-filled Elevation Map')
        fig5.colorbar(surf5, ax=ax5, shrink=0.5)
    else:
        ax5.text(0.5, 0.5, 0.5, 'No Data', transform=ax5.transAxes, ha='center', va='center')
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_zlabel('Elevation (m)')
    set_view_angle(ax5)  # Set view angle
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, '5_nan_filled_elevation_map.png'))
    # plt.show()
    plt.close(fig5)
    
    # Figure 6: Box-blur filtered elevation map
    fig6 = plt.figure(figsize=(10, 8), dpi=300)
    ax6 = fig6.add_subplot(111, projection='3d')
    if boxblur_map is not None:
        surf6 = ax6.plot_surface(xq, yq, boxblur_map, cmap='jet', edgecolor='none')
        ax6.set_title('Box-blur Filtered Elevation Map')
        fig6.colorbar(surf6, ax=ax6, shrink=0.5)
    else:
        ax6.text(0.5, 0.5, 0.5, 'No Data', transform=ax6.transAxes, ha='center', va='center')
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_zlabel('Elevation (m)')
    set_view_angle(ax6)  # Set view angle
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, '6_box_blur_filtered_elevation_map.png'))
    # plt.show()
    plt.close(fig6)
    
    # Figure 7: Bilateral filtered elevation map
    fig7 = plt.figure(figsize=(10, 8), dpi=300)
    ax7 = fig7.add_subplot(111, projection='3d')
    if bilateral_map is not None:
        surf7 = ax7.plot_surface(xq, yq, bilateral_map, cmap='jet', edgecolor='none')
        ax7.set_title('Bilateral Filtered Elevation Map')
        fig7.colorbar(surf7, ax=ax7, shrink=0.5)
    else:
        ax7.text(0.5, 0.5, 0.5, 'No Data', transform=ax7.transAxes, ha='center', va='center')
    ax7.set_xlabel('X (m)')
    ax7.set_ylabel('Y (m)')
    ax7.set_zlabel('Elevation (m)')
    set_view_angle(ax7)  # Set view angle
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, '7_bilateral_filtered_elevation_map.png'))
    # plt.show()
    plt.close(fig7)
    
    # Figure 8: Gaussian filtered elevation map
    fig8 = plt.figure(figsize=(10, 8), dpi=300)
    ax8 = fig8.add_subplot(111, projection='3d')
    if gaussian_map is not None:
        surf8 = ax8.plot_surface(xq, yq, gaussian_map, cmap='jet', edgecolor='none')
        ax8.set_title('Gaussian Filtered Elevation Map')
        fig8.colorbar(surf8, ax=ax8, shrink=0.5)
    else:
        ax8.text(0.5, 0.5, 0.5, 'No Data', transform=ax8.transAxes, ha='center', va='center')
    ax8.set_xlabel('X (m)')
    ax8.set_ylabel('Y (m)')
    ax8.set_zlabel('Elevation (m)')
    set_view_angle(ax8)  # Set view angle
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, '8_gaussian_filtered_elevation_map.png'))
    # plt.show()
    plt.close(fig8)


def main():
    # Load point cloud
    filename = 'assets/room.ply'  # Replace with your actual file
    original_point_cloud = o3d.io.read_point_cloud(filename)
    
    # Process point cloud
    processed_points = point_cloud_process_direct(original_point_cloud, grid_size=0.1, max_height=1.5, max_top_points_in_grid=3)
    
    # Build elevation map using direct statistical approach
    mean_map, std_map, xq, yq = build_elevation_map_direct(processed_points, resolution=0.1)
    
    # Fill NaN values with minimum of surrounding cells
    if mean_map is not None:
        filled_map = fill_nan_with_minimum_neighbors(mean_map)
        
        # Apply filters to the filled map
        print("Applying box-blur filter...")
        boxblur_map = boxblur_filter(filled_map)
        print("Applying Gaussian filter...")
        gaussian_map = gaussian_filter_func(filled_map)
        print("Applying bilateral filter...")
        bilateral_map = bilateral_filter(filled_map)
        
        # Create visualization of all maps
        visualize_all_maps(mean_map, std_map, xq, yq, original_point_cloud, processed_points,
                          filled_map, bilateral_map, boxblur_map, gaussian_map)
        print('Created all 8 elevation map figures, saved under assets folder, and displayed.')
    else:
        print("No elevation map data available.")


if __name__ == "__main__":
    main()