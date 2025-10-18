#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Use non-interactive backend to avoid Qt issues
# matplotlib.use('Agg')

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
            map_data = data.reshape((rows, cols))
            
        return map_data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def create_coordinate_grids(map_data, resolution=0.1):
    """
    Create coordinate grids for plotting.
    
    Parameters:
    map_data: 2D numpy array with the map data
    resolution: Grid resolution in meters
    
    Returns:
    xq, yq: Meshgrid coordinates
    """
    rows, cols = map_data.shape
    x_range = np.arange(0, cols * resolution, resolution)
    y_range = np.arange(0, rows * resolution, resolution)
    xq, yq = np.meshgrid(x_range[:cols], y_range[:rows])
    return xq, yq

def visualize_map(map_data, title, filename, xq, yq, colormap='jet'):
    """
    Create a 3D visualization of a map and save it to a file.
    
    Parameters:
    map_data: 2D numpy array with the map data
    title: Title for the plot
    filename: Output filename
    xq, yq: Coordinate grids
    colormap: Colormap to use
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if map_data is not None and not np.isnan(map_data).all():
        # Mask NaN values for plotting
        masked_data = np.ma.masked_where(np.isnan(map_data), map_data)
        surf = ax.plot_surface(xq, yq, masked_data, cmap=colormap, edgecolor='none')
        ax.set_title(title)
        fig.colorbar(surf, ax=ax, shrink=0.5)
    else:
        ax.text(0.5, 0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center')
        ax.set_title(title)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Value')
    
    # Set view angle
    ax.view_init(elev=70, azim=90)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved {filename}")

def main():
    # Ensure temp directory exists
    if not os.path.exists('temp/figure/maps_direct_static_cpp'):
        print("Error: maps directory not found. Run the C++ test program first.")
        return
    
    # Ensure output directory exists
    output_dir = 'temp/figure/maps_direct_static_cpp'
    os.makedirs(output_dir, exist_ok=True)
    
    # List of map names to process
    map_names = [
        'elevation',
        'uncertainty', 
        'slope',
        'step_height',
        'roughness',
        'traversability',
        'elevation_filtered'
    ]
    
    # Load and visualize each map
    elevation_map = None
    xq, yq = None, None
    
    for map_name in map_names:
        filepath = f'temp/binary/maps_direct_static/{map_name}.bin'
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            continue
            
        print(f"Loading {map_name} map...")
        map_data = load_binary_map(filepath)
        
        if map_data is None:
            print(f"Failed to load {map_name} map")
            continue
            
        # Store elevation map for coordinate grid creation
        if map_name == 'elevation' and elevation_map is None:
            elevation_map = map_data
            xq, yq = create_coordinate_grids(map_data)
            
        # Create visualization
        title = f'{map_name.replace("_", " ").title()} Map'
        output_file = os.path.join(output_dir, f'{map_name}_map.png')
        visualize_map(map_data, title, output_file, xq, yq)
    
    print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()