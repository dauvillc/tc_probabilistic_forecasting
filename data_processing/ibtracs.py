"""
Clément Dauvilliers - 2023 10 17
Implements functions to manipulate IBTrACS data.
"""

import pandas as pd


def load_ibtracs_data(path="data/IBTrACS/ibtracs_preprocessed.csv"):
    """
    Loads the preprocessed IBTrACS data from the given path.
    
    Parameters
    ----------
    path : str, optional
        The path to the preprocessed IBTrACS data file.

    Returns
    -------
    ibtracs_dataset : pandas.DataFrame.
    """
    ibtracs_dataset = pd.read_csv(path, parse_dates=["ISO_TIME"])
    return ibtracs_dataset
