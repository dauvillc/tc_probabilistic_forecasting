"""
Implements various utilities.
"""
import numpy as np
import pandas as pd
import torch


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
