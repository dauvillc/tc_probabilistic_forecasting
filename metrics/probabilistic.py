"""
Implements functions to evaluate a probabilistic forecast.
"""
import numpy as np
import pandas as pd
from utils.utils import to_numpy


def mae_per_threshold(y_true, predicted_params, inverse_CDF, thresholds,
                      y_true_quantiles=None, **kwargs):
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
    y_true_quantiles: list of floats, optional
        If not None, list of quantiles that define subsets of the data. The MAE per threshold
        is then computed for each subset of the data.
        For example, [0.9] will only compute the MAE per threshold for the 10% most extreme values.
    
    Keyword arguments
    -----------------
    **kwargs: dict
        Additional keyword arguments to be passed to inverse_CDF.

    Returns
    -------
    A Pandas DataFrame with columns "threshold", "MAE" and "MAE_q" for each quantile q in
    y_true_quantiles.
    """
    y_true = to_numpy(y_true)
    predicted_params = to_numpy(predicted_params)
    # Reshape the true values to (N * T,) and the predicted parameters to (N * T, P)
    # to evaluate all the samples and time steps at once
    y_true = y_true.reshape(-1)
    predicted_params = predicted_params.reshape(-1, predicted_params.shape[-1])
    thresholds = to_numpy(thresholds)

    # First: compute the MAE per threshold for the whole dataset
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
    # Create a DataFrame with the results
    df = pd.DataFrame({"threshold": thresholds, "MAE": all_maes})

    # Second: compute the MAE per threshold for each quantile
    if y_true_quantiles is not None:
        # Compute the quantiles of the true values
        y_true_quant_values = np.quantile(y_true, y_true_quantiles)
        # Create a DataFrame with the results
        for tau, q in zip(y_true_quantiles, y_true_quant_values):
            # Select the samples that above the quantile q
            q_idx = np.where(y_true >=  q)[0]
            # Compute the MAE per threshold
            q_maes = []
            for threshold in thresholds:
                mae = []
                for i in q_idx:
                    # Compute the quantiles from the predicted parameters
                    y_pred = inverse_CDF(predicted_params[i], threshold, **kwargs)
                    # Compute the MAE
                    mae.append(np.abs(y_true[i] - y_pred))
                q_maes.append(np.mean(mae))
            df[f"MAE_{tau}"] = q_maes
    return df

