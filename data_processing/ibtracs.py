"""
Clément Dauvilliers - 2023 10 17
Implements functions to manipulate IBTrACS data.
"""

import pandas as pd
import yaml


def load_ibtracs_data(path=None):
    """
    Loads the preprocessed IBTrACS data from the given path.
    
    Parameters
    ----------
    path : str, optional
        The path to the preprocessed IBTrACS data file.
        Defaults to the path specified in config.yaml.

    Returns
    -------
    ibtracs_dataset : pandas.DataFrame.
    """
    # Read the path from config.yaml if not specified
    if path is None:
        with open("config.yml") as file:
            config = yaml.safe_load(file)
        path = config['paths']["ibtracs_preprocessed"]
    ibtracs_dataset = pd.read_csv(path, parse_dates=["ISO_TIME"])
    return ibtracs_dataset
