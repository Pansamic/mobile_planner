#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

def load_config(config_path='config/config.yaml'):
    """
    Load configuration from YAML file.
    
    Parameters:
    config_path: Path to the config file
    
    Returns:
    config: Dictionary with configuration parameters
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Warning: Could not load config file {config_path}: {e}")
        return None

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
        if config is not None:
            resolution = config['grid_map']['resolution']
        else:
            resolution = 0.1  # Default resolution
    
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

def load_binary_path(file_path):
    """
    Load a binary path file.
    
    Binary format:
    - First 4 bytes: int32, number of waypoints
    - Remaining bytes: pairs of float32, (x, y) coordinates
    """
    with open(file_path, 'rb') as f:
        # Read number of waypoints
        num_waypoints = int.from_bytes(f.read(4), byteorder='little')
        
        # Read waypoints
        data = np.frombuffer(f.read(), dtype=np.float32)
        
        # Reshape to pairs of coordinates
        waypoints = data.reshape((num_waypoints, 2))
        
    return waypoints

def visualize_map_path(map_data, path_data, output_path):
    """
    Visualize map and path, and save to output file.
    """
    # Create coordinate grids
    xq, yq = create_coordinate_grids(map_data)
    
    # Rotate matrix 90 degrees clockwise
    map_data = np.rot90(map_data, k=1, axes=(1, 0))
    
    # Flip matrix upside down
    map_data = np.flipud(map_data)
    
    figsize = calculate_figsize(xq, yq)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display map
    # Mask NaN values for visualization
    if map_data is not None and not np.isnan(map_data).all():
        masked_map = np.ma.masked_where(np.isnan(map_data), map_data)
        im = ax.pcolormesh(xq, yq, masked_map, cmap='viridis', shading='auto')
        ax.set_title('Traversability Map with Path')
        fig.colorbar(im, ax=ax, label='Traversability')
    else:
        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Traversability Map with Path')
    
    # Overlay path if provided
    if path_data is not None and len(path_data) > 0:
        # Plot path
        ax.plot(path_data[:, 0], path_data[:, 1], 'r-', linewidth=2, marker='o', markersize=4)
        
        # Mark start and goal
        ax.plot(path_data[0, 0], path_data[0, 1], 'go', markersize=10, label='Start')
        ax.plot(path_data[-1, 0], path_data[-1, 1], 'mo', markersize=10, label='Goal')
        
        ax.legend()
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to '{output_path}'")

def print_help():
    """
    Print help information.
    """
    print("Usage: python visualize_map_path.py <map_file> <path_file> <output_image>")
    print("")
    print("Arguments:")
    print("  map_file       Path to binary map file")
    print("  path_file      Path to binary path file")
    print("  output_image   Path to output image file")
    print("")
    print("Example:")
    print("  python visualize_map_path.py temp/binary/maps_direct_static/traversability.bin temp/binary/maps_direct_static/waypoints.bin output.png")

def main():
    # Check command line arguments
    if len(sys.argv) != 4:
        print("Error: Incorrect number of arguments.")
        print("")
        print_help()
        sys.exit(1)
    
    map_file = sys.argv[1]
    path_file = sys.argv[2]
    output_image = sys.argv[3]
    
    # Check if files exist
    if not os.path.exists(map_file):
        print(f"Error: Map file '{map_file}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(path_file):
        print(f"Error: Path file '{path_file}' does not exist.")
        sys.exit(1)
    
    try:
        # Load map and path data
        map_data = load_binary_map(map_file)
        path_data = load_binary_path(path_file)
        
        if map_data is None:
            print(f"Error: Failed to load map data from '{map_file}'")
            sys.exit(1)
            
        # Visualize and save
        visualize_map_path(map_data, path_data, output_image)
        
        print(f"Visualization saved to '{output_image}'")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()