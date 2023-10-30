"""
Clément Dauvilliers - 2023 10 17
Implements functions to load various datasets.
"""

import os
import pandas as pd
import xarray as xr
import yaml
from tqdm import tqdm
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


def load_hursat_b1(storms_dataset, verbose=True):
    """
    Loads the HURSAT-B1 dataset, from the path specified in config.yml.

    Parameters
    ----------
    storms_dataset : pandas DataFrame
        Dataset of storms that includes the columns (SID, ISO_TIME). The function
        will search for the HURSAT-B1 data for each storm in this dataset.
    verbose : bool, optional
        Whether to print the number of storms found in the HURSAT-B1 data, and
        provide a progress bar.
    
    Returns
    -------
    found_storms: pandas DataFrame
        Dataset of storms that includes the columns (SID, ISO_TIME). Indicates
        which storms and timestamps were found in the HURSAT-B1 data.
    hursat_b1_dataset : xarray.DataArray
        DataArray of dimensions (sid_time, lat, lon) containing the HURSAT-B1
        data for the storms found in the storms_dataset.
    """
    if verbose:
        print("Loading HURSAT-B1 data...")
    # Read the path from config.yaml
    with open("config.yml") as file:
        config = yaml.safe_load(file)
    path = config['paths']["hursat_b1_preprocessed"]
    # Isolate the unique SIDs
    sids = storms_dataset['SID'].unique()
    # For each unique sid, check if there is a file in the HURSAT-B1 data
    # for it.
    found_storms = []
    hursat_b1_dataset = []
    iterator = tqdm(sids) if verbose else sids
    for sid in iterator:
        # The data is stored as path/year/sid.nc            
        # where year is the year the storm started in.
        year = storms_dataset[storms_dataset['SID'] == sid]['ISO_TIME'].dt.year.min()
        # Check if the file exists
        if os.path.exists(os.path.join(path, str(year), sid + ".nc")):
            # Load the file
            hursat_b1_dataset.append(xr.open_dataarray(os.path.join(path, str(year), sid + ".nc")))
            # Check which timestamps are in the file
            timestamps = hursat_b1_dataset[-1]['time'].values
            # Add the timestamps to the found_storms dataset
            found_storms.append(pd.DataFrame({"SID": [sid] * len(timestamps),
                                              "ISO_TIME": timestamps}))
    # Concatenate the found_storms dataset
    found_storms = pd.concat(found_storms, ignore_index=True)
    # Concatenate the hursat_b1_dataset
    hursat_b1_dataset = xr.concat(hursat_b1_dataset, dim='time')
    # Rename the time dimension to sid_time, as it actually now contains the successive
    # timestamps for each storm (same dimension as the ibtracs data dimension 0).
    hursat_b1_dataset = hursat_b1_dataset.rename({'time': 'sid_time'})
    # Print the number of storms found
    if verbose:
        print(f"Found {len(found_storms)}/{len(storms_dataset)} storms in the HURSAT-B1 data.")
        print(f"Dataset memory footprint: {hursat_b1_dataset.nbytes / 1e9} GB.")
    return found_storms, hursat_b1_dataset

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
