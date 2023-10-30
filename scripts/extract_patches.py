"""
Clément Dauvilliers - INRIA ARCHES - 2023-09-22
Extracts patches around the storms' centers from netcdf weather states
such as ERA5 or PanguWeather predictions.
This script is based on the preprocessed IBTrACS dataset, so make sure
to run scripts/preprocess_ibtracs.py first.
"""
import sys
sys.path.append('./')
import numpy as np
import pandas as pd
import xarray as xr
import os
import argparse
import time
import yaml
from tqdm import tqdm
from data_processing.datasets import load_ibtracs_data


def find_nearest(array, val):
    """
    Finds the nearest element to a given value within an array.
    :param array: array to search.
    :param val: value to compare with.
    """
    index = np.abs(array - val).argmin()
    return array[index]


def extract_patches(vartype, args):
    """
    Extracts patches around the storms' centers from netcdf weather states, and saves them to a netcdf file.
    This function virtually represents the main() function of this script, and avoids writing the same code twice
    for the atmospheric and surface variables.

    Parameters
    ----------
    vartype : str
        Type of variable to extract. Can be either 'atmo' or 'surface'.
    args : argparse.Namespace
        Arguments passed to the script.
    """
    patch_size = args.patch_size
    spatial_res = args.res
    suffix = args.suffix
    year = args.year + 2000
    # Read the configuration file
    with open("config.yml", 'r') as stream:
        config = yaml.safe_load(stream)
        # Input and output directories
        input_dir = config['paths']['era5']
        output_dir = config['paths']['era5_patches']
        # Variables to extract
        atmo_vars = config['era5']['atmo_variables']
        surface_vars = config['era5']['surface_variables']
        # Atmospheric pressure levels to extract
        pressure_levels = config['era5']['pressure_levels']
    # If there are no variables to extract, exit
    if len(atmo_vars) == 0 and vartype == 'atmo':
        print("No atmospheric variables to extract.")
        return
    if len(surface_vars) == 0 and vartype == 'surface':
        print("No surface variables to extract. Exiting.")
        return
    # If extracting atmospheric variables and no pressure levels are specified, exit
    if len(pressure_levels) == 0 and vartype == 'atmo':
        print("No pressure levels specified. Exiting.")
        return

    # Loads the preprocessed IBTrACS data
    storms_data = load_ibtracs_data()

    # Retrieves the first occurence of each storm
    storm_initial_times = storms_data[['SID', 'ISO_TIME']].groupby('SID').first().rename(columns={'ISO_TIME': 'FIRST_MONTH'}).reset_index()
    # For each first occurence, ignores the exact day and time by setting to the 01T0000 (e.g. 21st Jan 2018 12:34 becomes 2018/01/01 00:00)
    storm_initial_times['FIRST_MONTH'] = storm_initial_times['FIRST_MONTH'].dt.normalize().apply(lambda dt: dt.replace(day=1))
    # If a year is specified, only keep the storms that begun that year
    if year is not None:
        storm_initial_times = storm_initial_times[storm_initial_times['FIRST_MONTH'].dt.year == year]

    # For each month:
    #   Retrieve the ERA5 dataset for that month and the following one
    #      (as some storms span over two successive months).
    #   For each storm s that started during that month:
    #     Retrieve all entries for that storm in the IBTrACS data;
    #     For each such time t extract a patch from the full ERA5 image at time t
    #      around the storm's center.
    #   Stack the patches just extracted
    #   Save the dataset for that month to a netCDF file.

    for month in storm_initial_times['FIRST_MONTH'].unique():
        print("Processing month ", month)
        processing_time = time.time()

        # Load the dataset for the current month
        curr_dataset = xr.open_dataset(os.path.join(input_dir, f"{month.year}_{month.month:02}_{suffix}_{vartype}.nc"))
        # Load the dataset for the next month. xarray works on ncdf files lazily, so it will only be loaded
        # if there actually is a storm spanning over both months.
        next_month = month + pd.DateOffset(months=1)
        next_dataset = xr.open_dataset(os.path.join(input_dir, f"{next_month.year}_{next_month.month:02}_{suffix}_{vartype}.nc"))
        
        # Select the variables to extract
        if vartype == 'atmo':
            curr_dataset = curr_dataset[atmo_vars]
            next_dataset = next_dataset[atmo_vars]
            # Select the pressure levels to extract
            curr_dataset = curr_dataset.sel(level=pressure_levels)
            next_dataset = next_dataset.sel(level=pressure_levels)
        elif vartype == 'surface':
            curr_dataset = curr_dataset[surface_vars]
            next_dataset = next_dataset[surface_vars]

        storm_patches = []
        # The following will save the SID associated with every patch found during that month
        # (in particular, a SID can and will usually appear several successive times).
        patches_sids = []
        # Retrieves the SIDs of all storms that begun that month
        storm_ids = storm_initial_times[storm_initial_times['FIRST_MONTH'] == month]['SID']

        # For each storm that begun that month, extract the patches
        # and append them to the current's month lists
        for storm_id in tqdm(storm_ids):
            for row in storms_data[storms_data['SID'] == storm_id].itertuples(index=False):
                # Check whether the time point is in the current or the next month
                if row.ISO_TIME.month == month.month:
                    dataset = curr_dataset
                else:
                    dataset = next_dataset

                # Find the lat/lon point that is closest to the storm's center
                center_lon = find_nearest(dataset.coords['longitude'].data, row.LON)
                center_lat = find_nearest(dataset.coords['latitude'].data, row.LAT)
                # Define the area to extract
                area_lon = np.arange(center_lon - patch_size * spatial_res, center_lon + (patch_size + 1) * spatial_res, spatial_res)
                area_lat = np.arange(center_lat + patch_size * spatial_res, center_lat - (patch_size + 1) * spatial_res, -spatial_res)
                # If the center is close to the Greenwich meroidian, the area's longitude may be negative or beyond 360
                area_lon = area_lon % 360
                # Extract a square patch in each direction around the mean location of the storm
                patch = dataset.sel(time=row.ISO_TIME, longitude=area_lon, latitude=area_lat)

                # Since the patches cover different ranges of latitudes / longitudes, they cannot be coherently stacked.
                # The following Replaces the "longitude" and "latitude" dimensions with "pixel offsets"
                # dimensions (offset in pixels between a location and the patch's center), which are coherent across all patches.
                patch = patch.rename_dims({'longitude': 'h_pixel_offset', 'latitude': 'v_pixel_offset'})
                patch = patch.drop_vars(['longitude', 'latitude'])
                relative_coords = np.arange(-patch_size, patch_size + 1)
                patch = patch.assign_coords(v_pixel_offset=('v_pixel_offset', relative_coords),
                                                      h_pixel_offset=('h_pixel_offset', relative_coords))

                # Some variables contain (extremely) rare missing values (of the order of 1 every 20 years of data
                # and 69 variables). The following replaces those missing values with the mean of the variable
                # over the patch.
                for var in patch.data_vars:
                    if patch[var].isnull().any().data == False:
                        patch[var] = patch[var].fillna(patch[var].mean())
                
                # Save the patches to be latter written into that month's ncdf file
                storm_patches.append(patch)
                # Save the SID associated with that patch
                patches_sids.append(storm_id)

        # Stack all patches for that month, along a new dimension.
        month_dataset = xr.concat(storm_patches, dim="sid_time")
        # That dimension will be indexed by (time, sid). The "time" coordinate is already in the dataset,
        # the following adds the "sid" coordinate. Creating a MultiIndex after loading the file will be required
        # to actually select data based on a (time, sid) coordinate.
        month_dataset = month_dataset.assign_coords(sid=("sid_time", patches_sids))
        # Write the resulting dataset to a netcdf file
        month_dataset.to_netcdf(os.path.join(output_dir, month.strftime("%Y_%m") + f"_{vartype}_patches.nc"))
        print("Processed month in ", time.time() - processing_time, 's')


if __name__ == "__main__":
    # ======= ARGUMENT PARSING =======
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch_size', action='store', type=int, default=5,
            help="Number of pixels to include in every direction (NSEW) around the storm's center. Default to 5 pixels (sides of 11 pixels).") 
    parser.add_argument('--res', action='store', type=float, default=0.25,
            help="Spatial resolution in degrees - defaults to 0.25°")
    parser.add_argument('-s', '--suffix', action='store', default='pangu',
            help="Suffix of the input files - defaults to 'pangu'")
    parser.add_argument('-y', '--year', action='store', type=int, default=None,
            help="If specified, only extract patches for storms that begun that year. Defaults to None (extract patches for all storms).")
    args = parser.parse_args()
    
    # Extract atmospheric variables
    extract_patches('atmo', args)
    # Extract surface variables
    extract_patches('surface', args)
