"""
Implements various utilities.
"""
import numpy as np
import pandas as pd
import torch
import collections


def to_numpy(tensor):
    """
    Converts a tensor or array-like to a numpy array.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.type(torch.float32)
        return tensor.numpy(force=True)
    else:
        return np.array(tensor)


def daterange(start_date, end_date, step):
    """
    Yields a range of dates.

    Parameters
    ----------
    start_date : datetime.date
        The first date of the range.
    end_date : datetime.date
        The last date of the range.
    step : datetime.timedelta
        The step between two dates.
    """
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += step


def hours_to_sincos(times):
    """
    Converts an array-like of times to an array-like of sin/cos encoding.
    Ignores the date and only uses the hours.

    Parameters
    ----------
    times : array-like of datetime.datetime
        The times to convert. Only the hours are considered.

    Returns
    -------
    An array of shape (len(times), 2), containing the sin/cos encoding of the times.
    """
    # Convert the times to a pandas Series
    times = pd.Series(times, dtype='datetime64[ns]')
    # Ignore the date and only keep the hours
    times = times.dt.hour
    # Convert back to a numpy array
    times = times.values
    # Convert the hours to radians
    times = (times / 24) * 2 * np.pi
    # Compute the sin/cos encoding
    times = np.stack([np.sin(times), np.cos(times)], axis=1)
    return times


def matplotlib_markers(num):
    """
    Returns a list of matplotlib markers, which can be used to plot num lines
    and cycles through the markers.

    Parameters
    ----------
    num : int
        The number of markers to return.
    """
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd']
    return [markers[i % len(markers)] for i in range(num)]


def sshs_category(wind_speed):
    """
    Returns the Saffir-Simpson Hurricane Wind Scale category for a given wind speed.
    The categories go from -1 (Tropical Depression) to 5.

    Parameters
    ----------
    wind_speed : torch.Tensor of shape (N,)
        Batch of wind speeds, in knots.

    Returns
    -------
    torch.Tensor of shape (N,)
        The category of the wind speed.
    """
    # Define the thresholds for each category
    thresholds = torch.tensor([0, 34, 64, 83, 96, 113, 137, 220])
    # Compute the category (which goes from -1 to 5)
    return torch.bucketize(wind_speed, thresholds, right=True) - 2


def update_dict(d, u):
    """
    Updates a dictionary recursively, not in-place.
    Taken from https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    d = d.copy()
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def add_batch_dim(x, predictions):
    """
    Adds a batch dimension to an input x and to the predictions.

    Parameters
    ----------
    x : torch.Tensor of shape (T,) or (N, T)
        The input tensor.
    predictions : torch.Tensor of shape (T, C) or (N, T, C)
        The predictions tensor.

    Returns
    -------
    x : torch.Tensor of shape (1, T) or (N, T)
        The input tensor with a batch dimension.
    predictions : torch.Tensor of shape (1, T, C) or (N, T, C)
        The predictions tensor with a batch dimension.
        If predictions was of shape (T, C), it is replicated N times.
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if predictions.ndim == 2:
        predictions = predictions.unsqueeze(0).repeat(x.shape[0], 1, 1)
    return x, predictions


def average_score(score, reduction):
    """
    Returns the average of a score over a certain dimension.

    Parameters
    ----------
    score: torch.Tensor of shape (N, T, P) or (N, T)
        The score to average.
    reduction: str
        Which dimension(s) to average over. Can be "all", "samples", "time" or "none".
    
    Returns
    -------
    torch.Tensor of shape (0,) or (T,) or (N,) or (N, T)
        The average score.
    """
    if reduction == "all":
        return score.mean()
    elif reduction == "samples":
        return score.mean(dim=0)
    elif reduction == "time":
        return score.mean(dim=1)
    elif reduction == "none":
        return score
    else:
        raise ValueError("reduction must be 'all', 'samples', 'time' or 'none")
