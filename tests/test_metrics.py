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
    thresholds = np.array([0, 0.25, 0.5, 0.75, 100])
    mae = mae_per_threshold(y_true, predicted_quantiles, inverse_CDF, thresholds)
    print("Quantile regression:")
    print(mae)
    # Expected output:  [30. 30. 0. 0. 70.] (70 as the CDF is 1 only for values >= 100)

