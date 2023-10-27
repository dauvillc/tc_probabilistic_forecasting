"""
Clément Dauvilliers - 2023 10 17
Implements functions to load various datasets.
"""

import os
import pandas as pd
import xarray as xr
import yaml
from utils.datacube import select_sid_time


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


def load_era5_patches(storms_dataset, load_atmo=True, load_surface=True):
    """
    Loads the ERA5 patches for a given set of storms, from the directory specified
    in config.yml.
    The patches must have been extracted using scripts/extract_patches.py
    first.

    Parameters
    ----------
    storms_dataset : pandas DataFrame
        Dataset of storms that includes the columns (SID, ISO_TIME).
    load_atmo : bool, optional
        Whether to load the atmospheric patches. The default is True.
    load_surface : bool, optional
        Whether to load the surface patches. The default is True.
    
    Returns
    -------
    atmo_patches: xarray Dataset of dimensions
        (pressure level, horizontal_offset, vertical_offset, sid_time),
        containing the atmospheric fields.
        If load_atmo is False, returns None.
    surface_patches: xarray Dataset of dimensions
        (horizontal_offset, vertical_offset, sid_time),
        containing the surface fields.
        If load_surface is False, returns None.
    """
    # Retrieve the directory path from config.yml
    with open("config.yml") as file:
        config = yaml.safe_load(file)
    path = config['paths']["era5_patches"]

    atmo_patches, surface_patches = None, None

    # Load the atmospheric and surface patches if requested
    if load_atmo:
        atmo_patches = xr.open_mfdataset(os.path.join(path, "*_atmo_patches.nc"),
                                         combine="nested", concat_dim="sid_time",
                                         mask_and_scale=False)
        # Select the patches corresponding to the storms in storms_dataset
        atmo_patches = select_sid_time(atmo_patches, storms_dataset['SID'], storms_dataset['ISO_TIME'])
    # Load the surface patches if requested
    if load_surface:
        surface_patches = xr.open_mfdataset(os.path.join(path, "*_surface_patches.nc"),
                                            combine="nested", concat_dim="sid_time",
                                            mask_and_scale=False)
        surface_patches = select_sid_time(surface_patches, storms_dataset['SID'], storms_dataset['ISO_TIME'])

    return atmo_patches, surface_patches
