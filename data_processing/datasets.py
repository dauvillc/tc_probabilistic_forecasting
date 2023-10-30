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


def load_hursat_b1(storms_dataset, use_cache=True, verbose=True):
    """
    Loads the HURSAT-B1 dataset, from the path specified in config.yml.

    Parameters
    ----------
    storms_dataset : pandas DataFrame
        Dataset of storms that includes the columns (SID, ISO_TIME). The function
        will search for the HURSAT-B1 data for each pair (SID, ISO_TIME) in this dataset.
    use_cache : bool, optional
        Whether to use a cached version of the HURSAT-B1 data. If False, the
        function will load the data from the original files. The default is True.
    verbose : bool, optional
        Whether to print the number of storms found in the HURSAT-B1 data, and
        provide a progress bar.
    
    Returns
    -------
    found_storms: pandas DataFrame
        Dataset of storms that includes the columns (SID, ISO_TIME). Indicates
        which storms and timestamps were found in the HURSAT-B1 data.
    hursat_b1_dataset : xarray.DataArray
        DataArray of dimensions (time, lat, lon) containing the HURSAT-B1
        data for the storms found in the storms_dataset.
    """
    if verbose:
        print("Loading HURSAT-B1 data...")
    # Load the config file
    with open("config.yml") as file:
        config = yaml.safe_load(file)
    path_cache = config['paths']["hursat_b1_cache"]
    # If use_cache is True, load the cached version
    if use_cache:
        # Check if the cache exists
        if not os.path.exists(path_cache):
            print('Cache not found. Loading from original files...')
            return load_hursat_b1(storms_dataset, use_cache=False, verbose=verbose)
        print("Using cached version.")
        hursat_b1_dataset = xr.open_dataarray(path_cache).set_index(sid_time=['sid', 'time'])
        found_storms = hursat_b1_dataset.sid_time.to_pandas().reset_index()[['sid', 'time']]
        # Rename to SID and ISO_TIME to be coherent with the IBTrACS dataset
        found_storms = found_storms.rename(columns={'sid': 'SID', 'time': 'ISO_TIME'})
    else:
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
                # Load the file. We need to load it as a dataset of one variable
                # instead of a DataArray, because "time" is both a dimension and a coordinates,
                # and a DataArray cannot rename one of them without renaming the other.... (please fix).
                dataset = xr.open_dataset(os.path.join(path, str(year), sid + ".nc"))
                # Rename the "time" dimension to "sid_time"
                dataset = dataset.rename_dims({'time': 'sid_time'})
                # Add a coordinate "sid" with the current sid to the "sid_time" dimension
                sids = [sid] * len(dataset['sid_time'])
                dataset = dataset.assign_coords(sid=('sid_time', sids))
                # Create a MultiIndex to index the sid_time dimension by pairs (sid, time)
                dataset = dataset.set_index(sid_time=['sid', 'time'])
                # Retrieve the rows of storms_dataset that correspond to the current sid
                # and timestamps by joining on the sid and ISO_TIME columns.
                sid_times = dataset.sid_time.to_pandas().rename('sid_time')
                sid_rows = storms_dataset.merge(sid_times,
                                                left_on=['SID', 'ISO_TIME'],
                                                right_index=True)[['SID', 'ISO_TIME']]
                # Select within the hursat_b1_dataset the timestamps that were found
                dataset = select_sid_time(dataset, sid_rows['SID'], sid_rows['ISO_TIME'])
                # Select the only variable in the dataset to obtain a DataArray
                dataset = dataset['IRWIN']
                # Transform the dimension "time" into "sid_time", whih is a multiindex
                # e.g. dataset.sel(sid_time=(sid, time))
                hursat_b1_dataset.append(dataset)
                found_storms.append(sid_rows)
        # Concatenate the found_storms dataset
        found_storms = pd.concat(found_storms, ignore_index=True)
        # Concatenate the hursat_b1_dataset
        hursat_b1_dataset = xr.concat(hursat_b1_dataset, dim='sid_time')
        # Write the dataset to the cache
        hursat_b1_dataset.reset_index(['sid_time']).to_netcdf(path_cache)
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
