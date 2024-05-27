import numpy as np
from scipy.optimize import minimize

# An example to demonstrate how to use nelder mead algorithm to find the optimal weights for
# the models in an ensemble 

# Dummy data
np.random.seed(42)
predictions = np.random.rand(100, 3)  # 100 data points, predictions from 3 models
target = np.random.rand(100)  # 100 actual target values

def rmse_func(weights):
    pred = (predictions * weights).sum(axis=1)
    rmse = np.sqrt(1 / len(pred) * ((target - pred)**2).sum())
    return rmse

n_models = predictions.shape[1]
# Start by giving equal weight to each model ( = 1 / n_models). Sum of weights is 1.
initial_weights = np.ones(n_models) / n_models

# We want to find the set of weights that minimizes the RMSE. We start with the initial weights.
res = minimize(rmse_func, initial_weights, method='Nelder-Mead')

weights = res["x"]
rmse = res["fun"]

print("Optimal Weights:", weights)
print("Optimal RMSE:", rmse)
