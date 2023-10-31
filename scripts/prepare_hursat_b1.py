"""
Must be run after downloading the HURSAT B1 dataset, via
scripts/download_hursat_b1.py.

Prepares the HURSAT B1 dataset:
    - For each storm in the HURSAT B1 downloaded data, read the data every
        6 hours, and save it in a separate file. If several satellites have
        taken measurements at the same time, use only one based on a predefined order.
        Stack the images at each time step into a single netCDF file.
    - Create a CSV file containing the list of all storms that were found, which can be
        merged with the IBTrACS data.
"""
import sys
sys.path.append("./")
import os
import yaml
import xarray as xr
import argparse
import pandas as pd
from tqdm import tqdm
from utils.datacube import set_relative_spatial_coords, upscale_and_crop


def extract_time_from_filename(filename):
    """
    Extracts the time from a HURSAT B1 file name.
    The files are named as follows:
    SID.NAME.YYYY.MM.DD.HHmm.ANGLE.SAT.WIND.hursat-b1.version.nc
    where ANGLE is the satellite's zenith angle, SAT is the satellite's abreviation,
    and WIND is the approximate wind speed in knots.
    See https://www.ncei.noaa.gov/products/hurricane-satellite-data
    for more information.
    """
    timestamp = f"{filename.split('.')[2]}-{filename.split('.')[3]}-{filename.split('.')[4]}T{filename.split('.')[5]}"
    timestamp = pd.to_datetime(timestamp)
    return timestamp


def load_hourly_snapshots(year, sid, res_hours=6, crop_size=None):
    """
    Loads the hourly snapshots of a specific storm, and assembles
    them into a single xarray DataArray.
    When several satellites are available, selects the one that is nadirmost regarding
    the storm's center (i.e. the satellite's zenith point is closest).
    
    Parameters
    ----------
    year : int
        The year of the storm.
    sid : str
        The storm ID.
    res_hours : int, optional
        The time resolution of the snapshots, in hours. Default is 6.
    crop_size : int, optional
        The size of the central crop, in pixels. Default is None, which means no crop.
    
    Returns
    -------
    xarray.DataArray of dimensions (time, lat, lon) or None
        The hourly snapshots of the storm. If for some timestep, no data was found
        from any satellite, returns None.
    """
    # Path to the storm's directory
    storm_dir = os.path.join(hursat_b1_path, str(year), sid)
    # List all files in the storm's directory.
    files = os.listdir(storm_dir)
    # Retrieve all iso timestamps from the files and conert them to datetime objects
    timestamps = [extract_time_from_filename(f) for f in files]
    # Remove duplicates and sort the timestamps
    timestamps = sorted(list(set(timestamps)))
    # Keep only the timestamps that are multiples of the time resolution
    timestamps = [t for t in timestamps if t.hour % res_hours == 0]
    # For each timestamp, find the file with the lowest satellite zenith angle.
    # Load the corresponding file and add it to the list of snapshots.
    # For some rare timestamps, the satellite data is fully missing. In this case,
    # we try to use another satellite that does not have optimal viewing conditions.
    snapshots = []
    for timestamp in timestamps:
        # Browse all satellite files for the current timestamp
        # and order them by increasing satellite zenith angle
        files_for_timestamp = [f for f in files if extract_time_from_filename(f) == timestamp]
        files_for_timestamp = sorted(files_for_timestamp, key=lambda f: float(f.split('.')[6]))
        # Try the successive files until finding one that does not have NaN values
        snapshot = None
        for file in files_for_timestamp:
            # Load the file
            snapshot = xr.open_dataset(os.path.join(storm_dir, file))['IRWIN']
            # Upscale the snapshot to a resolution from its native resolution of 0.07°
            # to 0.25°, to match the resolution of ERA5.
            # At the same time, crop a central patch.
            snapshot = upscale_and_crop(snapshot, dims=('lat', 'lon'), new_res=0.25, crop_size=crop_size)
            # If the snapshot does not contain any NaN values, keep it
            # Once again, the vast majority of snapshots are valid, so this loop
            # will only be executed a few times.
            if not snapshot.isnull().any():
                break
            snapshot = None
        if snapshot is None:
            # Return None if no valid snapshot was found, discard the storm
            return None
        snapshots.append(snapshot)
         
    # At this point, we cannot concatenate the snapshots because they have different
    # values of latitude and longitude. We need to set the relative spatial coordinates
    # that will be the same for each snapshot.
    snapshots = [set_relative_spatial_coords(snap, lat_dim="lat", lon_dim="lon")
                 for snap in snapshots]

    # Stack the snapshots along the time dimension
    snapshots = xr.concat(snapshots, dim='htime')
    # Rename the htime dimension to time
    snapshots = snapshots.rename({'htime': 'time'})

    # Finally: the timestamps do not correspond to exact hours as in the IBTrACS data
    # (e.g. 23:59:599999 instead of 00:00:00). We need to round them to the nearest hour.
    snapshots['time'] = snapshots['time'].dt.round('H')

    return snapshots


if __name__ == "__main__":
    # Arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", "-s",  type=int, default=2000,
                        help="The first year to consider.")
    parser.add_argument("--end_year", "-e", type=int, default=2021,
                        help="The last year to consider.")
    parser.add_argument("--crop_size", "-c", type=int, default=None,
                        help="The size of the central crop, in pixels.")
    args = parser.parse_args()
    start_year, end_year = args.start_year + 2000, args.end_year + 2000

    # Paths extraction
    with open("config.yml") as file:
        config = yaml.safe_load(file)
    hursat_b1_path = config['paths']["hursat_b1"]
    hursat_b1_preprocessed_path = config['paths']["hursat_b1_preprocessed"]

    # For each year, for each storm, load the hourly snapshots and assemble them
    # into a single xarray DataArray. Save the DataArray in a netCDF file.
    for year in range(start_year, end_year + 1):
        print(f"Processing year {year}...")
        # Path to the directory containing the storms of the current year
        year_dir = os.path.join(hursat_b1_path, str(year))
        # Create the output directory if it does not exist
        output_dir = os.path.join(hursat_b1_preprocessed_path, str(year))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # List of all storms in the current year in the HURSAT data
        storms = os.listdir(year_dir)
        for sid in tqdm(storms):
            # This will also crop central patches, if crop_size was specified
            snapshots = load_hourly_snapshots(year, sid, crop_size=args.crop_size)
            # Save the snapshots in a netCDF file
            if snapshots is not None:
                snapshots.to_netcdf(os.path.join(output_dir, f"{sid}.nc"))
