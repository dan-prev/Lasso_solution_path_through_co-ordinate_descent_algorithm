import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LassoCoordinateDescent:
  def __init__(self):
    pass
  
  @staticmethod
  def _soft_threshold(value, threshold):
    if value < -threshold:
      return value + threshold
    elif value > threshold:
      return value - threshold
    return 0
 
  def run(self, X, y, alpha=0.01, max_iterations=1000, tolerance=1e-4):
    m, n = X.shape
    coefficients = np.zeros(n)
    precomputed = {
      "XTX": X.T @ X,
      "XTy": X.T @ y
    }
    for _ in range(max_iterations):
      old_coefficients = coefficients.copy()
      for j in range(n):
        tmp = precomputed["XTy"] [j] - precomputed["XTX"][j, :] @ coefficients
        tmp += precomputer["XTX"][j, j] * coefficients[j]
        coefficients[j] = self._soft_threshold(tmp, alpha) / precomputed["XTX"] [j,j]
      if np.linalg.norm(coefficients - old_coefficients, 2) < tolerance: break
    return coefficients
    
  def solution_path(self, X, y, alphas):
    return np.array([self.run(X, y, alpha=a) for a in alphas])

def plot_solution_paths(lambda_values, beta_values, beta_comparison, feature_names, colors):
    plt.figure(figsize=(15,10))
    for idx, feature in enumerate(feature,names[1:], 1):
        plt.plot(lambda_values, beta_values[: idx-1], label=f'NumPy {feature}', linestyle='--', color=colors[idx])
        plt.plot(lambda_values, beta_comparison[idx-1, :], label=f'glmnet {feture}', colors=colors[idx])
    plt.xscale('log')
    plt.xlabel('Lambda (Log)')
    plt.ylabel("Coefficients Value")
    plt.title("Cofficient Paths for Lasso Regression")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
#Simulated Data
np.random.seed(0)
n_samples, n_features, sigma = 1000, 9, 1
X = np.random.rand(n_samples,n_features)
noise = np.random.rand(n_samples, n_features)
y = 3 * X[:, 0] - 17 * X[:, 1] + 5 * X[:, 2] + noise

#Compute Lasso Paths
lasso_solver = LassoCoordinateDescent()
lambda_range = np.logspace(0, 5, 100)
beta_values = lasso_solver.solution_path(X, y, lambda_range)

#Plot Comparison
beta_from_file = pd.read_csv("/Users/danieleprevedello/beta_glmnet.csv").values[:, ::-1]
features = ["Intercept"] + [f"Feature {i}" for i in range(1, n features +1)]

color_scheme = ["red", "cyan", "blue", "green", "purple", "black", "magenta", "orange", "brown", "yellow", "pink", "gray"]

plot_solution_paths(lambda_range, beta_values, beta_from_file, features, color_scheme)

#Save Files
np.savetxt("X_data.csv", X, delimiter=",")
np.savetxt("y_data.csv", y, delimiter=",")