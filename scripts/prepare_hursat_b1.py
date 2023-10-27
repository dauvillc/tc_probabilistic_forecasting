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
from tqdm import tqdm
from utils.datacube import set_relative_spatial_coords


def load_hourly_snapshots(year, sid):
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
    
    Returns
    -------
    xarray.DataArray of dimensions (time, lat, lon)
        The hourly snapshots of the storm.
    """
    # Path to the storm's directory
    storm_dir = os.path.join(hursat_b1_path, str(year), sid)
    # List all files in the storm's directory. Their names are of the form
    # SID.NAME.YYYY.MM.DD.HHmm.ANGLE.SAT.WIND.hursat-b1.version.nc
    # where ANGLE is the satellite's zenith angle, SAT is the satellite's abreviation,
    # and WIND is the approximate wind speed in knots.
    # See https://www.ncei.noaa.gov/products/hurricane-satellite-data
    # for more information.
    files = os.listdir(storm_dir)
    # Retrieve all iso timestamps from the files and convert them to datetime objects
    timestamps = ['.'.join(f.split(".")[2:6]) for f in files]
    # Remove duplicates and sort the timestamps
    timestamps = sorted(list(set(timestamps)))
    # For each timestamp, find the file with the lowest satellite zenith angle.
    # Load the corresponding file and add it to the list of snapshots.
    snapshots = []
    for timestamp in timestamps:
        # Find the file with the lowest satellite zenith angle
        min_angle = 91
        min_file = None
        for f in files:
            if '.'.join(f.split(".")[2:6]) == timestamp:
                angle = float(f.split(".")[6])
                if angle < min_angle:
                    min_angle = angle
                    min_file = f
        # Load the file
        snapshots.append(xr.open_dataset(os.path.join(storm_dir, min_file))['IRWIN'])
    
    # At this point, we cannot concatenate the snapshots because they have different
    # values of latitude and longitude. We need to set the relative spatial coordinates
    # that will be the same for each snapshot.
    snapshots = [set_relative_spatial_coords(snap, lat_dim="lat", lon_dim="lon")
                 for snap in snapshots]

    # Stack the snapshots along the time dimension
    snapshots = xr.concat(snapshots, dim='htime')
    # Rename the htime dimension to time
    snapshots = snapshots.rename({'htime': 'time'})
    return snapshots


if __name__ == "__main__":
    # Arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", "-s",  type=int, default=2000,
                        help="The first year to consider.")
    parser.add_argument("--end_year", "-e", type=int, default=2021,
                        help="The last year to consider.")
    args = parser.parse_args()
    start_year, end_year = args.start_year, args.end_year

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
            snapshots = load_hourly_snapshots(year, sid)
            # Save the snapshots in a netCDF file
            snapshots.to_netcdf(os.path.join(output_dir, f"{sid}.nc"))
