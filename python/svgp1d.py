import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def original_function(x):
    """Define the original function to model"""
    return 3 * np.sin(0.1 * x) ** 2 + np.log(x + 1)


def squared_distance(A, B):
    """
    Computes pairwise squared Euclidean distance between two sets of vectors.
    Replaces the builtin pdist2 function.
    Input: A is M-by-D, B is N-by-D
    Output: D2 is M-by-N
    """
    A = np.array(A)
    B = np.array(B)
    
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
        
    if A.shape[1] != B.shape[1]:
        raise ValueError('Inputs must have the same number of columns')
    
    # This calculation relies on the identity: ||a - b||^2 = ||a||^2 - 2a'b + ||b||^2
    # It is implemented efficiently using matrix operations.
    A_sq = np.sum(A ** 2, axis=1, keepdims=True)
    B_sq = np.sum(B ** 2, axis=1, keepdims=True)
    D2 = A_sq - 2 * np.dot(A, B.T) + B_sq.T
    return D2


def kernel(x1, x2, l, sigma_f):
    """Squared Exponential Kernel function"""
    D2 = squared_distance(x1, x2)
    return sigma_f ** 2 * np.exp(-0.5 * D2 / l ** 2)


def negative_elbo(params, X, y, M):
    """
    Extracts parameters and computes the negative ELBO, the objective to be minimized.
    """
    # Unpack parameters
    N = X.shape[0]
    hyperparams = params[:3]
    Z = params[3:].reshape(-1, 1)
    
    l = np.exp(hyperparams[0])
    sigma_f = np.exp(hyperparams[1])
    sigma_n = np.exp(hyperparams[2])
    
    # Calculate required Kernel matrices
    K_mm = kernel(Z, Z, l, sigma_f) + 1e-6 * np.eye(M)  # Add jitter
    K_mn = kernel(Z, X, l, sigma_f)
    
    # Calculate the ELBO
    # Formula: L = log N(y|0, Q_nn + sigma_n^2*I) - (1/(2*sigma_n^2)) * trace(K_nn - Q_nn)
    # Where Q_nn = K_nm * K_mm^-1 * K_mn
    
    L_m = np.linalg.cholesky(K_mm)
    
    A = (1.0 / sigma_n) * np.linalg.solve(L_m, K_mn)
    AAT = np.dot(A, A.T)
    B = AAT + np.eye(M)
    L_B = np.linalg.cholesky(B)
    
    log_det_term = -N/2 * np.log(2*np.pi) - np.sum(np.log(np.diag(L_B))) - (N-M)/2*np.log(sigma_n**2)
    
    c = np.linalg.solve(L_B.T, np.linalg.solve(L_B, np.dot(A, y)))
    quad_term = -0.5 * (np.dot(y.T, y)/sigma_n**2 - np.dot(c.T, c))
    
    log_likelihood = log_det_term + quad_term
    
    K_nn_diag_sum = N * sigma_f**2
    Q_nn_diag = np.sum(np.linalg.solve(L_m, K_mn)**2, axis=0)
    trace_term = -0.5 * (1/sigma_n**2) * (K_nn_diag_sum - np.sum(Q_nn_diag))
    
    elbo = log_likelihood + trace_term
    
    return -elbo


def svgp1d():
    """Sparse Gaussian Process with Variational Inference Implementation"""
    
    # 1. Define the Data
    num_data = 5000
    x_range = [0, 100]
    noise_std = 0.5
    
    x_train = np.linspace(x_range[0], x_range[1], num_data).reshape(-1, 1)
    y_train = original_function(x_train.ravel()) + noise_std * np.random.randn(num_data)
    
    # 2. Initialize GP Parameters and Inducing Points
    num_inducing_points = 50  # M
    
    # Initialize inducing points 'Z' by selecting them uniformly from the training set
    inducing_indices = np.round(np.linspace(0, num_data-1, num_inducing_points)).astype(int)
    Z_initial = x_train[inducing_indices]
    
    # Initialize hyperparameters [log(length_scale), log(signal_variance), log(noise_variance)]
    # We optimize logs for numerical stability and to ensure they remain positive.
    log_l = np.log(50.0)
    log_sigma_f = np.log(2.0)
    log_sigma_n = np.log(noise_std)
    initial_hyperparams = np.array([log_l, log_sigma_f, log_sigma_n])
    
    # Combine all parameters into a single vector for the optimizer.
    # The parameters are: [hyperparams; inducing_point_locations]
    initial_params = np.concatenate([initial_hyperparams, Z_initial.ravel()])
    
    # 3. Optimize Parameters by Minimizing the Negative ELBO
    print('Optimizing hyperparameters and inducing points...')
    
    # Define the objective function
    objective_function = lambda params: negative_elbo(params, x_train, y_train, num_inducing_points)
    
    # Run the optimizer
    result = minimize(objective_function, initial_params, method='L-BFGS-B')
    
    print('Optimization finished.')
    
    # Extract the optimized parameters
    optimal_params = result.x
    optimal_hyperparams = optimal_params[:3]
    Z_optimal = optimal_params[3:].reshape(-1, 1)
    
    l = np.exp(optimal_hyperparams[0])
    sigma_f = np.exp(optimal_hyperparams[1])
    sigma_n = np.exp(optimal_hyperparams[2])
    
    # 4. Make Predictions with Optimized Parameters
    print('Making predictions with optimized model...')
    
    # Center the training data mean (optional but good practice)
    mean_y = np.mean(y_train)
    y_train_centered = y_train - mean_y
    
    # Create a dense set of points for plotting
    x_plot = np.linspace(x_range[0] - 20, x_range[1] + 20, 1000).reshape(-1, 1)
    
    # Build Kernel matrices using the OPTIMIZED inducing points (Z_optimal)
    K_mm = kernel(Z_optimal, Z_optimal, l, sigma_f) + 1e-6 * np.eye(num_inducing_points)  # Add jitter for stability
    K_sm = kernel(x_plot, Z_optimal, l, sigma_f)
    K_ss = kernel(x_plot, x_plot, l, sigma_f)  # For variance, we just need the diagonal of this
    
    # The predictive mean and variance formulas for the VFE model
    K_mn = kernel(Z_optimal, x_train, l, sigma_f)
    A = K_mm + (1/sigma_n**2) * np.dot(K_mn, K_mn.T)
    
    # NOTE: Cholesky decomposition
    L_A = np.linalg.cholesky(A)
    B = (1/sigma_n**2) * np.dot(K_mn, y_train_centered)
    
    # NOTE: Solving a linear system with a Cholesky factor
    alpha = np.linalg.solve(L_A.T, np.linalg.solve(L_A, B))
    mu_pred_plot = np.dot(K_sm, alpha) + mean_y
    
    # Predictive variance calculation
    v = np.linalg.solve(L_A, K_sm.T)
    var_pred_plot = np.diag(K_ss) - np.sum(v**2, axis=0)
    
    sd_pred_plot = np.sqrt(var_pred_plot)
    y_ci_plot = np.column_stack([mu_pred_plot - 1.96 * sd_pred_plot, 
                                 mu_pred_plot + 1.96 * sd_pred_plot])
    
    # 5. Visualize the Results
    plt.figure(figsize=(12, 8))
    
    # Plot confidence interval
    plt.fill_between(x_plot.ravel(), y_ci_plot[:, 0], y_ci_plot[:, 1], 
                     color=[0.8, 0.85, 1.0], alpha=0.5, label='95% Confidence Interval')
    
    # Plot subset of training data
    plt.scatter(x_train[::50], y_train[::50], c=[0.7, 0.7, 0.7], s=20, 
                label='Subset of Training Data', alpha=0.7)
    
    # Plot sparse GP mean
    plt.plot(x_plot, mu_pred_plot, 'b-', linewidth=2, label='Sparse GP Mean')
    
    # Plot optimized inducing points
    Z_y = original_function(Z_optimal.ravel())
    plt.scatter(Z_optimal, Z_y, c='red', s=50, marker='o', 
                label='Optimized Inducing Points', zorder=5)
    
    plt.title('Sparse GP with Variational Inference (VFE)')
    plt.xlabel('Input (x)')
    plt.ylabel('Output (y)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    svgp1d()