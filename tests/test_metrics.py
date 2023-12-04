"""
Tests the data loading and evaluation functions.
"""
import sys
sys.path.append("./")
import numpy as np
from metrics.probabilistic import mae_per_threshold
from metrics.quantiles import Quantiles_inverse_eCDF


if __name__ == "__main__":
    # Create a fake dataset with 1 sample and 1 time steps 
    y_true = np.array([[30]])

    # Quantile regression
    # -------------------
    quantiles = np.array([0.5])
    # Create a fake set of predicted quantiles for each time step 
    predicted_quantiles = np.array([[[30]]])
    # Create the inverse empirical CDF function
    inverse_CDF = Quantiles_inverse_eCDF(quantiles, min_val=0, max_val=100)
    # Compute the MAE per threshold
    thresholds = np.array([0, 0.25, 0.5, 0.75, 1])
    mae = mae_per_threshold(y_true, predicted_quantiles, inverse_CDF, thresholds)
    print("Quantile regression:")
    print(mae)
    # Expected output:
    # Quantile regression:
    #    threshold  MAE
    # 0       0.00  30.0
    # 1       0.25  30.0
    # 2       0.50   0.0
    # 3       0.75   0.0
    # 4       1.00  70.0

