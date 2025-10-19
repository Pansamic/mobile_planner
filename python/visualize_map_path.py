#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import os

def load_binary_map(file_path):
    """
    Load a binary map file.
    
    Binary format:
    - First 4 bytes: int32, number of rows
    - Next 4 bytes: int32, number of columns
    - Remaining bytes: float32, matrix data (row-major order)
    """
    with open(file_path, 'rb') as f:
        # Read dimensions
        rows = int.from_bytes(f.read(4), byteorder='little')
        cols = int.from_bytes(f.read(4), byteorder='little')
        
        # Read data
        data = np.frombuffer(f.read(), dtype=np.float32)
        
        # Reshape to matrix
        matrix = data.reshape((cols, rows))
        
    return matrix.T

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
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Display map
    # Mask NaN values for visualization
    masked_map = np.ma.masked_where(np.isnan(map_data), map_data)
    im = ax.imshow(masked_map, cmap='viridis', origin='lower')
    
    # Overlay path if provided
    if path_data is not None and len(path_data) > 0:
        # Plot path
        ax.plot(path_data[:, 1], path_data[:, 0], 'r-', linewidth=2, marker='o', markersize=4)
        
        # Mark start and goal
        ax.plot(path_data[0, 1], path_data[0, 0], 'go', markersize=10, label='Start')
        ax.plot(path_data[-1, 1], path_data[-1, 0], 'mo', markersize=10, label='Goal')
        
        ax.legend()
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Traversability Map with Path')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Traversability')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

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
        
        # Visualize and save
        visualize_map_path(map_data, path_data, output_image)
        
        print(f"Visualization saved to '{output_image}'")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()