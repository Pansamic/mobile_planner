import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def original_function(x1, x2):
    """Define the true 2D function we are trying to model (sombrero function)"""
    r = np.sqrt(x1**2 + x2**2)
    # Handle the case where r is 0 to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(r == 0, 1, np.sin(r) / r)
    return result


def kernel(x1, x2, l=2.0, sigma_f=1.0):
    """
    Squared Exponential Kernel function for any number of dimensions.
    Calculates the covariance between two sets of points.
    """
    # Compute pairwise squared Euclidean distances
    x1 = np.array(x1)
    x2 = np.array(x2)
    
    # If 1D arrays, reshape to column vectors
    if x1.ndim == 1:
        x1 = x1.reshape(-1, 1)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)
        
    # Compute squared distances using broadcasting
    dist_matrix = np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** 2, axis=2)
    return sigma_f ** 2 * np.exp(-0.5 * dist_matrix / l ** 2)


def gp2d():
    """2D Gaussian Process Regression Implementation"""
    
    # 1. Define the Data
    # Generate a sparse grid of training data points from the function
    num_points_per_dim = 10  # This will create a 10x10 grid, 100 points total
    range_vals = [-8, 8]
    vec = np.linspace(range_vals[0], range_vals[1], num_points_per_dim)
    x1_train, x2_train = np.meshgrid(vec, vec)
    
    # Combine the grid points into an N x 2 matrix for the GP
    X_train = np.column_stack([x1_train.ravel(), x2_train.ravel()])
    
    # Calculate the output values and add some random noise
    y_train = original_function(X_train[:, 0], X_train[:, 1]) + 0.1 * np.random.randn(X_train.shape[0])
    
    # Center the training data around a zero mean
    mean_y = np.mean(y_train)
    y_train_centered = y_train - mean_y
    
    # 2. Define GP Parameters and Kernel Function
    l = 2.0           # Length scale
    sigma_f = 1.0     # Signal variance
    sigma_n = 0.2     # Noise variance
    
    # 3. Manual GP Prediction Calculations
    # Create a dense grid of points for plotting a smooth surface
    num_plot_points_per_dim = 50
    plot_vec = np.linspace(range_vals[0], range_vals[1], num_plot_points_per_dim)
    x1_plot, x2_plot = np.meshgrid(plot_vec, plot_vec)
    X_plot = np.column_stack([x1_plot.ravel(), x2_plot.ravel()])
    
    # Build the necessary Kernel (Covariance) matrices
    K = kernel(X_train, X_train, l, sigma_f)
    K_s = kernel(X_plot, X_train, l, sigma_f)
    K_ss = kernel(X_plot, X_plot, l, sigma_f)
    
    # Add noise to the training covariance matrix: K + sigma_n^2 * I
    Ky = K + sigma_n ** 2 * np.eye(X_train.shape[0])
    
    # Calculate the posterior mean using the core GP equation
    mu_pred_plot_vec = K_s.dot(np.linalg.solve(Ky, y_train_centered))
    mu_pred_plot_vec = mu_pred_plot_vec + mean_y  # Add the mean back
    
    # Calculate the posterior variance
    cov_pred_plot = K_ss - K_s.dot(np.linalg.solve(Ky, K_s.T))
    var_pred_plot_vec = np.diag(cov_pred_plot)
    sd_pred_plot_vec = np.sqrt(var_pred_plot_vec)
    
    # Reshape the prediction vectors back into a grid for surface plotting
    mu_pred_surf = mu_pred_plot_vec.reshape(num_plot_points_per_dim, num_plot_points_per_dim)
    sd_pred_surf = sd_pred_plot_vec.reshape(num_plot_points_per_dim, num_plot_points_per_dim)
    
    # 4. Visualize the Results
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle('2D Gaussian Process Regression')
    
    # Subplot 1: Predicted Mean Surface
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(x1_plot, x2_plot, mu_pred_surf, alpha=0.8, cmap='viridis', edgecolor='none')
    
    # Plot training data points on top of the surface
    ax1.scatter(X_train[:, 0], X_train[:, 1], y_train, c='red', s=50, alpha=1)
    
    ax1.set_title('GP Predicted Mean Surface')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('y')
    ax1.view_init(30, 25)
    
    # Add colorbar
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # Subplot 2: Uncertainty (Standard Deviation) Surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(x1_plot, x2_plot, sd_pred_surf, cmap='plasma', edgecolor='none')
    
    ax2.set_title('GP Prediction Uncertainty (Standard Deviation)')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('Std. Dev.')
    ax2.view_init(30, 25)
    
    # Add colorbar
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    gp2d()