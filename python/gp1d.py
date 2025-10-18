import numpy as np
import matplotlib.pyplot as plt


def original_function(x):
    """Define the original function to model - Pure square wave"""
    # Creating a pure square wave function with period of 4 units
    x_mod = np.mod(x, 4)  # Get the position within each period
    
    # Initialize output array
    y = np.zeros_like(x)
    
    # High level (first 2 units of period)
    high_mask = x_mod < 2
    y[high_mask] = 2.0
    
    # Low level (next 2 units of period)
    low_mask = (x_mod >= 2) & (x_mod < 4)
    y[low_mask] = -1.0
    
    return y

def original_function(x):
    """Define the original function to model"""
    # Creating a periodic function with sine and square wave sections
    # Period is 4 units: 2 units of sine wave followed by 2 units of square wave
    x_mod = np.mod(x, 4)  # Get the position within each period
    
    # Initialize output array
    y = np.zeros_like(x)
    
    # Sine wave section (first 2 units of period)
    sine_mask = x_mod < 2
    y[sine_mask] = 2 * np.sin(np.pi * x_mod[sine_mask])
    
    # Square wave section (next 2 units of period)
    square_mask = (x_mod >= 2) & (x_mod < 4)
    y[square_mask] = np.where(x_mod[square_mask] < 3, 1.5, -1.5)
    
    return y

def original_function(x):
    """Define the original function to model - Previous implementation"""
    return 3 * np.sin(0.1 * x) ** 2 + np.log(x + 1)


def kernel(x1, x2, l=1.5, sigma_f=4.8):
    """
    Squared Exponential Kernel function.
    Calculates the covariance between two sets of points.
    """
    # Compute pairwise squared distances
    x1 = np.array(x1).reshape(-1, 1)
    x2 = np.array(x2).reshape(-1, 1)
    dist_matrix = np.sum((x1[:, np.newaxis] - x2[np.newaxis, :]) ** 2, axis=2)
    return sigma_f ** 2 * np.exp(-0.5 * dist_matrix / l ** 2)


def gp1d():
    """1D Gaussian Process Regression Implementation"""
    
    # 1. Define the Data
    num_data = 1000
    x_range = [0, 200]
    
    x_train = np.linspace(x_range[0], x_range[1], num_data)
    y_train = original_function(x_train)
    
    # Specific points where you want predictions
    x_specific = np.random.uniform(x_range[0], x_range[1], num_data // 2)
    
    # Center the training data around a zero mean
    mean_y = np.mean(y_train)
    y_train_centered = y_train - mean_y
    
    # 2. Define GP Parameters
    l = 1.5           # Length scale: controls the smoothness of the function
    sigma_f = 4.8     # Signal variance: controls the vertical variation
    sigma_n = 1.0     # Noise variance: represents the noise in the observations
    
    # 3. Manual GP Prediction Calculations
    # Create a dense set of points for plotting a smooth curve
    x_plot = np.linspace(x_range[0] - 1, x_range[1] + 1, num_data * 10)
    
    # Build the necessary Kernel (Covariance) matrices
    K = kernel(x_train, x_train, l, sigma_f)
    K_s = kernel(x_plot, x_train, l, sigma_f)
    K_ss = kernel(x_plot, x_plot, l, sigma_f)
    
    # Add noise to the training covariance matrix
    Ky = K + sigma_n ** 2 * np.eye(len(x_train))
    
    # Calculate the posterior mean (the predictions)
    # Using np.linalg.solve is more numerically stable than matrix inverse
    mu_pred_plot = K_s.dot(np.linalg.solve(Ky, y_train_centered))
    
    # Calculate the posterior covariance
    cov_pred_plot = K_ss - K_s.dot(np.linalg.solve(Ky, K_s.T))
    
    # Get the variance (diagonal of the covariance matrix) and standard deviation
    var_pred_plot = np.diag(cov_pred_plot)
    sd_pred_plot = np.sqrt(var_pred_plot)
    
    # Add the original mean back to the predictions for plotting
    mu_pred_plot = mu_pred_plot + mean_y
    
    # Calculate the 95% confidence interval
    y_ci_plot = np.column_stack([mu_pred_plot - 1.96 * sd_pred_plot, 
                                 mu_pred_plot + 1.96 * sd_pred_plot])
    
    # Get predictions for specific points
    K_s_specific = kernel(x_specific, x_train, l, sigma_f)
    y_pred_specific = K_s_specific.dot(np.linalg.solve(Ky, y_train_centered))
    y_pred_specific = y_pred_specific + mean_y  # Add mean back
    
    # 4. Visualize the Results
    plt.figure(figsize=(12, 8))
    
    # Plot the 95% confidence interval as a shaded area
    plt.fill_between(x_plot, y_ci_plot[:, 0], y_ci_plot[:, 1], 
                     color=[0.8, 0.85, 1.0], alpha=0.5, label='95% Confidence Interval')
    
    # Plot the GP's mean prediction as a solid line
    plt.plot(x_plot, mu_pred_plot, 'b-', linewidth=2, label='GP Mean Prediction')
    
    # Plot the original training data points
    plt.scatter(x_train[::10], y_train[::10], c='red', s=50, 
                marker='o', label='Training Data', zorder=5)
    
    # Plot the specific predictions as 'x' markers
    plt.scatter(x_specific[::10], y_pred_specific[::10], c='black', s=80, 
                marker='x', linewidth=2, label='Specific Predictions', zorder=5)
    
    # Add plot formatting
    plt.title('1D Gaussian Process Regression (Manual Calculation)')
    plt.xlabel('Input (x)')
    plt.ylabel('Output (y)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 5. Display Numerical Results
    print('\n--- Gaussian Process Predictions (Manual) ---')
    results = np.column_stack([x_specific, y_pred_specific])
    print('Input_x\t\tPredicted_y')
    for i in range(min(10, len(x_specific))):  # Print first 10 results
        print(f'{results[i, 0]:.3f}\t\t{results[i, 1]:.3f}')


if __name__ == "__main__":
    gp1d()