"""
Implements functions to evaluate a probabilistic forecast.
"""
import numpy as np
from utils.utils import to_numpy


def mae_per_threshold(y_true, predicted_params, inverse_CDF, thresholds, **kwargs):
    """
    Computes the mean absolute error (MAE) per threshold.

    Parameters
    ----------
    y_true: array-like or torch.Tensor of shape (N, T)
        The true values.
    predicted_params: array-like or torch.Tensor of shape (N, T, P)
        where N is the number of samples, P is the number of parameters
        and T is the number of time steps.
        The predicted parameters, which characterize the predictive distribution.
    inverse_CDF: callable f: (predicted_params, u) -> ndarray
        Function that computes the inverse of the empirical CDF at a given probability u,
        given the predicted parameters.
    thresholds: array-like or torch.Tensor
        Probability thresholds at which the MAE is evaluated, as floats between 0 and 1.
    
    Keyword arguments
    -----------------
    **kwargs: dict
        Additional keyword arguments to be passed to inverse_CDF.

    Returns
    -------
    The MAE per threshold, as a ndarray of shape (len(thresholds),).
    """
    y_true = to_numpy(y_true)
    predicted_params = to_numpy(predicted_params)
    # Reshape the true values to (N * T,) and the predicted parameters to (N * T, P)
    # to evaluate all the samples and time steps at once
    y_true = y_true.reshape(-1)
    predicted_params = predicted_params.reshape(-1, predicted_params.shape[-1])
    thresholds = to_numpy(thresholds)
    # We cannot assume inverse_CDF to be vectorized, so we loop over the thresholds
    # and the samples
    all_maes = []
    for threshold in thresholds:
        mae = []
        for i in range(predicted_params.shape[0]):
            # Compute the quantiles from the predicted parameters
            y_pred = inverse_CDF(predicted_params[i], threshold, **kwargs)
            # Compute the MAE
            mae.append(np.abs(y_true[i] - y_pred))
        all_maes.append(np.mean(mae))
    return np.array(all_maes)

