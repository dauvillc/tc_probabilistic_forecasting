"""
Usage: python scripts/preprocess_tcir.py

This script preprocesses the TCIR dataset. The path to the dataset must be specified
in config.yml beforehand.
"""
import sys
sys.path.append('./')
import yaml
import xarray as xr
import numpy as np
import pandas as pd


if __name__ == '__main__':
    # Load config file 
    with open("config.yml", 'r') as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    _TCIR_PATH_ = cfg['paths']['tcir']

    # Load the tabular information
    data_info = pd.read_hdf(_TCIR_PATH_, key="info", mode='r')

    # Load the datacube
    datacube = xr.open_dataset(_TCIR_PATH_)['matrix']

    # === TABULAR DATA PREPROCESSING ===
    # Rename the columns to match IBTrACS
    data_info = data_info.rename({'data_set': 'BASIN',
                              'ID': 'SID',
                              'time': 'ISO_TIME',
                              'lat': 'LAT',
                              'lon': 'LON',
                              'Vmax': 'INTENSITY'
                             }, axis="columns")
    # Convert ISO_TIME to datetime
    data_info['ISO_TIME'] = pd.to_datetime(data_info['ISO_TIME'], format='%Y%m%d%H')

    # === TEMPORAL SELECTION ===
    # The temporal resolution in TCIR is 3 hours, but the best-track data's original resolution
    # (such as ATCF given by the JTWC) is 6 hours. One point out of two was linearly interpolated
    # (which is very common with best-track data), which would make the problem artificially easier.
    # To avoid this, we'll only keep the points that are multiples of 6 hours.
    # Select the points in the info
    data_info = data_info[data_info['ISO_TIME'].dt.hour.isin([0, 6, 12, 18])]
    # Select the corresponding entries in the datacube
    datacube = datacube.isel(phony_dim_4=data_info.index)
    # Reset the index of data_info so that it matches datacube.isel
    data_info = data_info.reset_index()

    # === DATACUBE PREPROCESSING ===
    # Rename the dimensions
    datacube = datacube.rename({"phony_dim_4": "sid_time",
                            "phony_dim_5": "v_pixel_offset",
                            "phony_dim_6": "h_pixel_offset",
                            "phony_dim_7": "variable"})
    # Remove the Water vapor and Visible channels, as supposedly do not
    # provide information that complements the IR channel.
    # Besides, the Visible channel is unstable due to daylight.
    datacube = datacube.drop_sel(variable=[1, 2])

    # Add coordinates for the latitude and longitude dimensions
    # Since the lat/lon differs for each storm and time, we'll use the offset
    # (in pixels) from the center of the image
    img_side_len = datacube.shape[1]
    offset = np.arange(-(img_side_len // 2), img_side_len // 2 + 1)
    datacube = datacube.assign_coords(h_pixel_offset=('h_pixel_offset', offset),
                                      v_pixel_offset=('v_pixel_offset', offset))

    # Add coordinates for the sid_time dimension
    # This coordinate will be in the format of a tuple (SID, ISO_TIME)
    datacube = datacube.assign_coords(sid=('sid_time', data_info['SID'].values),
                                  time=('sid_time', data_info['ISO_TIME'].values))

    # Finally, we'll add a coordinate for the variable dimension
    datacube = datacube.assign_coords(variable=('variable', ['IR', 'PMW']))
    
    # === OUTLIERS AND MISSING VALUES ==== (SEE NOTEBOOK FOR EXPLAINATIONS)
    # Convert the unreasonably large values to NaNs
    datacube = datacube.where(datacube[:, :, :, 1] < 10^3)
    # Compute the ratio of NaNs (native + converted from outliers) for each sample
    nan_counts = np.array([datacube[k].isnull().sum() for k in range(datacube.shape[0])])
    nan_ratios = nan_counts / (datacube.shape[1] * datacube.shape[2] * 2)
    
    # * For every storm S that contains at least one sample that is more than 10% NaN:
    #   * Find all segments of S's trajectory that do not contain any NaN;
    #   * Consider each of those segments as an independent trajectory, by giving them new SIDs.
    #   * Discard the samples containing NaNs (and that are thus fully NaN).
    # * Where the images contain less than 10% of NaNs, fill them with zeros.
    
    # Retrieve the index of the samples that are fully NaN.
    where_full_nan = np.where(nan_ratios >= 0.1)[0]
    # Retrieve the SIDs corresponding to those samples
    sids_with_full_nan = data_info.iloc[where_full_nan]['SID'].unique()

    for sid in sids_with_full_nan:
        segments = []
        current_segment = []
        currently_in_segment = False
        # Iterate through the samples of that storm. If we find a NaN, that's the end
        # of the current segment. Keep iterating until finding a non full-NaN sample,
        # which is the start of a new segment.
        for k in data_info[data_info['SID'] == sid].index:
            if k in where_full_nan:
                if currently_in_segment:
                    # Stop the current segment
                    segments.append(np.array(current_segment))
                    current_segment = []
                    currently_in_segment = False
            else:
                if not currently_in_segment:
                    # Start a new segment if not currently in a segment
                    currently_in_segment = True
                current_segment.append(k)

        sid_values = data_info['SID'].values.copy()
        for n_seg, seg in enumerate(segments):
            # Give all samples in that segment a new SID
            sid_values[seg] = sid_values[seg] + f'_{n_seg}'
        data_info['SID'] = sid_values
    # Drop the samples that are fully NaNs
    data_info = data_info.drop(index=where_full_nan)
    # Select the index of data_info (which doesn't contain the full-NaN samples anymore) in the datacube
    datacube = datacube.isel(sid_time=data_info.index)
    # Reset the index of the tabular data, so that it matches that of the datacube
    data_info = data_info.reset_index()
    # We can now interpolate the partially-NaN images
    datacube = datacube.interpolate_na('h_pixel_offset', method='nearest')
    # NaN values at the border won't be interpolated, we'll fill them with zeros.
    datacube = datacube.fillna(0)

    # === SAVE THE PREPROCESSED DATA ===
    # Retrieve the save path for the tabular data and the datacube from the config file
    _TCIR_INFO_PATH_ = cfg['paths']['tcir_info_preprocessed']
    _TCIR_DATACUBE_PATH_ = cfg['paths']['tcir_datacube_preprocessed']

    # Save the tabular data
    data_info.to_csv(_TCIR_INFO_PATH_, index=False)
    # Save the datacube
    datacube.to_netcdf(_TCIR_DATACUBE_PATH_)
