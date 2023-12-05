"""
Implements functions to evaluate a probabilistic forecast.
"""
import numpy as np
import pandas as pd
from utils.utils import to_numpy


def metric_per_threshold(metric, y_true, predicted_params, inverse_CDF, thresholds,
                         y_true_quantiles=None, **kwargs):
    """
    Given a metric L(y_true, y_pred) and a probabilistic forecast, computes the metric
    per threshold: Let F be the CDF of the predictive distribution, then the metric
    is computed as L(y_true, F^{-1}(u)) for each threshold u in thresholds.

    Parameters
    ----------
    metric: str
        Name of the metric to be evaluated. Can be "bias", "MAE" or "RMSE".
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
        Probability thresholds at which the metric is evaluated, as floats between 0 and 1.
    y_true_quantiles: list of floats, optional
        If not None, list of quantiles that define subsets of the data. The MAE per threshold
        is then computed for each subset of the data.
        For example, [0.9] will only compute the metric per threshold for the 10% most extreme values.
    
    Keyword arguments
    -----------------
    **kwargs: dict
        Additional keyword arguments to be passed to inverse_CDF.

    Returns
    -------
    A Pandas DataFrame with columns "threshold", "<metric>" and "<metric>_q" for each quantile q in
    y_true_quantiles, where <metric> is replaced by the metric name.
    """
    # Define the metric
    if metric == "bias":
        metric_func = lambda y_true, y_pred: y_pred - y_true
    elif metric == "mae":
        metric_func = lambda y_true, y_pred: np.abs(y_pred - y_true)
    elif metric == "rmse":
        metric_func = lambda y_true, y_pred: (y_pred - y_true)**2
    y_true = to_numpy(y_true)
    predicted_params = to_numpy(predicted_params)
    # Reshape the true values to (N * T,) and the predicted parameters to (N * T, P)
    # to evaluate all the samples and time steps at once
    y_true = y_true.reshape(-1)
    predicted_params = predicted_params.reshape(-1, predicted_params.shape[-1])
    thresholds = to_numpy(thresholds)

    # First: compute the metric per threshold for all the samples
    # We cannot assume inverse_CDF to be vectorized, so we loop over the thresholds
    # and the samples
    all_measures = []
    for threshold in thresholds:
        measure = []
        for i in range(predicted_params.shape[0]):
            # Compute the quantiles from the predicted parameters
            y_pred = inverse_CDF(predicted_params[i], threshold, **kwargs)
            # Compute the MAE
            measure.append(metric_func(y_true[i], y_pred))
        measure = np.mean(measure)
        if metric == "rmse":
            measure = np.sqrt(measure)
        all_measures.append(measure)
    # Create a DataFrame with the results
    df = pd.DataFrame({"threshold": thresholds, metric: all_measures})

    # Second: compute the metric per threshold for each quantile
    if y_true_quantiles is not None:
        # Compute the quantiles of the true values
        y_true_quant_values = np.quantile(y_true, y_true_quantiles)
        # Create a DataFrame with the results
        for tau, q in zip(y_true_quantiles, y_true_quant_values):
            # Select the samples that above the quantile q
            q_idx = np.where(y_true >=  q)[0]
            # Compute the metric  per threshold
            q_measures = []
            for threshold in thresholds:
                measure = []
                for i in q_idx:
                    # Compute the quantiles from the predicted parameters
                    y_pred = inverse_CDF(predicted_params[i], threshold, **kwargs)
                    # Compute the metric
                    measure.append(metric_func(y_true[i], y_pred))
                measure = np.mean(measure)
                if metric == "rmse":
                    measure = np.sqrt(measure)
                q_measures.append(measure)
            df[f"{metric}_{tau}"] = q_measures
    return df

