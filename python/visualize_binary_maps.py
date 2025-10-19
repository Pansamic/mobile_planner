#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import yaml
import glob

# Use non-interactive backend to avoid Qt issues
# matplotlib.use('Agg')


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


def load_binary_map(filepath):
    """
    Load a binary map from file.
    
    Parameters:
    filepath: Path to the binary file
    
    Returns:
    map_data: 2D numpy array with the map data
    """
    try:
        with open(filepath, 'rb') as f:
            # Read dimensions
            rows = int.from_bytes(f.read(4), byteorder='little')
            cols = int.from_bytes(f.read(4), byteorder='little')
            
            # Read data
            data = np.frombuffer(f.read(), dtype=np.float32)
            map_data = data.reshape((cols, rows))
            
        return map_data.T
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def create_coordinate_grids(map_data, resolution=None):
    """
    Create coordinate grids for plotting.
    
    Parameters:
    map_data: 2D numpy array with the map data
    resolution: Grid resolution in meters
    
    Returns:
    xq, yq: Meshgrid coordinates
    """
    # Load config if resolution is not provided
    if resolution is None:
        config = load_config()
        resolution = config['grid_map']['resolution']
    
    rows, cols = map_data.shape
    # Center the grid around the origin (0, 0)
    x_range = np.arange(-rows//2, rows//2) * resolution
    y_range = np.arange(-cols//2, cols//2) * resolution
    xq, yq = np.meshgrid(x_range, y_range)
    return xq, yq

def calculate_figsize(xq, yq):
    """
    Calculate appropriate figure size based on coordinate grid dimensions.
    
    Parameters:
    xq, yq: Coordinate grids
    
    Returns:
    figsize: Tuple of (width, height) for the figure
    """
    if xq.size == 0 or yq.size == 0:
        return (10, 8)
    
    # Calculate the range of each dimension
    x_range = np.ptp(xq)
    y_range = np.ptp(yq)
    
    # Calculate aspect ratio
    aspect_ratio = x_range / y_range if y_range != 0 else 1
    
    # Set base figure size
    base_width = 10
    base_height = 8
    
    # Adjust based on aspect ratio
    if aspect_ratio > 1:
        # Wider than tall
        width = base_width
        height = base_width / aspect_ratio
    else:
        # Taller than wide
        height = base_height
        width = base_height * aspect_ratio
    
    # Ensure minimum size
    width = max(width, 6)
    height = max(height, 6)
    
    return (width, height)

def visualize_map_3d(map_data, title, filename, xq, yq, colormap='jet'):
    """
    Create a 3D visualization of a map and save it to a file.
    
    Parameters:
    map_data: 2D numpy array with the map data
    title: Title for the plot
    filename: Output filename
    xq, yq: Coordinate grids
    colormap: Colormap to use
    """
    figsize = calculate_figsize(xq, yq)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if map_data is not None and not np.isnan(map_data).all():
        # Mask NaN values for plotting
        masked_data = np.ma.masked_where(np.isnan(map_data), map_data)
        surf = ax.plot_surface(xq, yq, masked_data, cmap=colormap, edgecolor='none')
        # Set aspect ratio only if coordinate grids are not empty
        if xq.size > 0 and yq.size > 0:
            ax.set_box_aspect((np.ptp(xq), np.ptp(yq), np.ptp(masked_data)))
    else:
        ax.text(0.5, 0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')
        ax.set_title(title)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Value')
    
    # Set view angle
    ax.view_init(elev=90, azim=90)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved {filename}")


def visualize_map_2d(map_data, title, filename, xq, yq, colormap='jet'):
    """
    Create a 2D visualization of a map and save it to a file.
    
    Parameters:
    map_data: 2D numpy array with the map data
    title: Title for the plot
    filename: Output filename
    xq, yq: Coordinate grids
    colormap: Colormap to use
    """
    figsize = calculate_figsize(xq, yq)
    fig, ax = plt.subplots(figsize=figsize)
    
    if map_data is not None and not np.isnan(map_data).all():
        # Mask NaN values for plotting
        masked_data = np.ma.masked_where(np.isnan(map_data), map_data)
        im = ax.pcolormesh(xq, yq, masked_data, cmap=colormap, shading='auto')
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')
        ax.set_title(title)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved {filename}")


def print_help():
    """Print help information."""
    help_text = """
Usage: python3 visualize_binary_maps.py <input> <output_directory>

Arguments:
    input              Path to a binary map file or directory containing binary maps
    output_directory   Directory where visualizations will be saved

Description:
    This script visualizes binary map files as both 2D and 3D plots.
    If input is a file, only that file will be visualized.
    If input is a directory, all valid binary maps in that directory will be visualized.
"""
    print(help_text)


def process_single_file(filepath, output_dir, resolution=None):
    """
    Process a single binary map file.
    
    Parameters:
    filepath: Path to the binary map file
    output_dir: Directory to save visualizations
    resolution: Grid resolution in meters
    """
    print(f"Loading map from {filepath}...")
    map_data = load_binary_map(filepath)
    
    if map_data is None:
        print(f"Failed to load map from {filepath}")
        return
    
    # Get filename without extension for title
    filename = os.path.splitext(os.path.basename(filepath))[0]
    title = f'{filename.replace("_", " ").title()} Map'
    
    # Create coordinate grids
    xq, yq = create_coordinate_grids(map_data, resolution)

    # rotate matrix 90 degrees clockwise
    map_data = np.rot90(map_data, k=1, axes=(1, 0))

    # up side down matrix
    map_data = np.flipud(map_data)
    
    # Create 3D visualization
    output_file_3d = os.path.join(output_dir, f'{filename}_3d.png')
    visualize_map_3d(map_data, title, output_file_3d, xq, yq)
    
    # Create 2D visualization
    output_file_2d = os.path.join(output_dir, f'{filename}_2d.png')
    visualize_map_2d(map_data, title, output_file_2d, xq, yq)


def process_directory(directory_path, output_dir, resolution=None):
    """
    Process all binary map files in a directory.
    
    Parameters:
    directory_path: Path to directory containing binary maps
    output_dir: Directory to save visualizations
    resolution: Grid resolution in meters
    """
    # Find all .bin files in the directory
    binary_files = glob.glob(os.path.join(directory_path, "*.bin"))
    
    if not binary_files:
        print(f"No binary map files found in {directory_path}")
        return
    
    print(f"Found {len(binary_files)} binary map files in {directory_path}")
    
    for filepath in binary_files:
        process_single_file(filepath, output_dir, resolution)


def main():
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    # Check command line arguments
    if len(sys.argv) != 3:
        print("Error: Incorrect number of arguments.")
        print_help()
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Check if input path exists
    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist.")
        print_help()
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    try:
        config = load_config()
        grid_resolution = config['grid_map']['resolution']
    except Exception as e:
        print(f"Warning: Could not load config file, using default resolution of 0.1")
        grid_resolution = 0.1
    
    # Process input based on whether it's a file or directory
    if os.path.isfile(input_path):
        # Process single file
        process_single_file(input_path, output_dir, grid_resolution)
    elif os.path.isdir(input_path):
        # Process directory
        process_directory(input_path, output_dir, grid_resolution)
    else:
        print(f"Error: Input path '{input_path}' is neither a file nor a directory.")
        print_help()
        sys.exit(1)
    
    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()