"""
Tests the implementation of the quantile CRPS.
"""
import sys
sys.path.append("./")
import numpy as np
from utils.metrics import QuantilesCRPS


if __name__ == "__main__":
    # Create some dummy data
    # Quantiles to predict, of shape (n_quantiles,)
    quantiles = np.array([0.25, 0.5, 0.75])
    # True observations, of shape (n_samples, time steps)
    # One of the observations is before the first quantile, one is in the
    # middle, and one is after the last quantile
    y_1 = np.array([[1]])
    y_2 = np.array([[5]])
    y_3 = np.array([[7]])
    # Predictions, of shape (n_samples, time steps, n_quantiles)
    y_pred = np.array([[[2, 4, 6]]])

    # Compute the CRPS
    crps = QuantilesCRPS(quantiles, 0, 8)
    print("CRPS for y_1:", crps(y_pred, y_1))
    print("CRPS for y_2:", crps(y_pred, y_2))
    print("CRPS for y_3:", crps(y_pred, y_3))
