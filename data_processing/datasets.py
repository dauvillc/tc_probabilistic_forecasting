"""
Clément Dauvilliers - 2023 10 17
Implements functions to load various datasets.
"""

import os
import pandas as pd
import xarray as xr
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

def load_era5_patches(start_date, end_date):
    """
    Loads the ERA5 patches from the directory specified
    in config.yml.
    The patches must have been extracted using scripts/extract_patches.py
    first.

    Parameters
    ----------
    start_date : datetime.datetime
        The start date of the patches to load.
    end_date : datetime.datetime
        The end date of the patches to load (included).
    
    Returns
    -------
    atmo_patches: xarray Dataset of dimensions
        (pressure level, horizontal_offset, vertical_offset, sid_time),
        containing the atmospheric fields.
    surface_patches: xarray Dataset of dimensions
        (horizontal_offset, vertical_offset, sid_time),
        containing the surface fields.
    """
    # Retrieve the directory path from config.yml
    with open("config.yml") as file:
        config = yaml.safe_load(file)
    path = config['paths']["era5_patches"]

    # Load the atmospheric patches, that are stored as monthly files,
    # and concatenate them along the sid_time dimension
    date_range = pd.date_range(start_date, end_date, freq="MS")
    atmo_patches, surface_patches = [], []
    for date in date_range:
        atmo_path = os.path.join(path, f"{date.year}_{date.month:02d}_atmo_patches.nc")
        surface_path = os.path.join(path, f"{date.year}_{date.month:02d}_surface_patches.nc")
        atmo = xr.open_dataset(atmo_path, mask_and_scale=False)
        surface = xr.open_dataset(surface_path, mask_and_scale=False)
        atmo_patches.append(atmo)
        surface_patches.append(surface)
    atmo_patches = xr.concat(atmo_patches, dim="sid_time")
    surface_patches = xr.concat(surface_patches, dim="sid_time")

    return atmo_patches, surface_patches
