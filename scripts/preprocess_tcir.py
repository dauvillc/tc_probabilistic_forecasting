"""
Usage: python scripts/preprocess_tcir.py

This script preprocesses the TCIR dataset. The path to the dataset must be specified
in config.yml beforehand.
"""

import sys

sys.path.append("./")
from pathlib import Path
import yaml
import xarray as xr
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from utils.utils import hours_to_sincos, months_to_sincos, sshs_category_array
from utils.preprocessing import grouped_shifts_and_deltas
from utils.train_test_split import stormwise_train_test_split


if __name__ == "__main__":
    # Load config file
    with open("config.yml", "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    # Paths to the TCIR dataset
    _TCIR_PATH_1_ = cfg["paths"]["tcir_atln"]  # Atlantic part
    _TCIR_PATH_2_ = cfg["paths"]["tcir_sh"]  # Southern Hemisphere part
    _TCIR_PATH_3_ = cfg["paths"]["tcir_2017"]  # 2017 part
    # Path to the ERA5 patches
    _ERA5_PATCHES_PATH_ = cfg["paths"]["era5_patches"]
    # Output directory
    save_dir = Path(cfg["paths"]["tcir_preprocessed_dir"])
    interm_cube_path = save_dir / "datacube_intermediate.nc"
    interm_info_path = save_dir / "info_intermediate.csv"

    # Check if the intermediate data is already saved
    if not (interm_cube_path.exists() and interm_info_path.exists()):
        print("Loading the raw TCIR dataset...")
        # Load the tabular information
        data_info = pd.concat(
            [
                pd.read_hdf(_TCIR_PATH_1_, key="info", mode="r"),
                pd.read_hdf(_TCIR_PATH_2_, key="info", mode="r"),
                pd.read_hdf(_TCIR_PATH_3_, key="info", mode="r"),
            ]
        ).reset_index(drop=True)

        # Load the datacubes and concatenate them at once
        datacube = xr.open_mfdataset(
            [_TCIR_PATH_1_, _TCIR_PATH_2_, _TCIR_PATH_3_],
            combine="nested",
            concat_dim="phony_dim_4",
            chunks={"phony_dim_4": 8},
        )["matrix"]

        # === TABULAR DATA PREPROCESSING ===
        # Rename the columns to match IBTrACS
        data_info = data_info.rename(
            {
                "data_set": "BASIN",
                "ID": "SID",
                "time": "ISO_TIME",
                "lat": "LAT",
                "lon": "LON",
                "Vmax": "INTENSITY",
            },
            axis="columns",
        )
        # Convert ISO_TIME to datetime
        data_info["ISO_TIME"] = pd.to_datetime(data_info["ISO_TIME"], format="%Y%m%d%H")

        # === OPTIONAL SUBSAMPLING ===
        # If the dataset is too large, we can subsample it by selecting a random subset of the
        # storms. This is done by specifying the 'tcir/subsample' key in the config file.
        if cfg["preprocessing"]["subsample"]:
            fraction = cfg["preprocessing"]["subsample_fraction"]
            # Select a random subset of the storms
            sids = data_info["SID"].unique()
            sids = np.random.default_rng(42).choice(
                sids, size=int(len(sids) * fraction), replace=False
            )
            data_info = data_info[data_info["SID"].isin(sids)]
            # Select the corresponding entries in the datacube
            datacube = datacube.isel(phony_dim_4=data_info.index)
            # Reset the index of data_info so that it matches datacube.isel
            data_info = data_info.reset_index(drop=True)
            print(f"Subsampled to {len(data_info['SID'].unique())} storms.")

        # === TEMPORAL SELECTION ===
        # The temporal resolution in TCIR is 3 hours, but the best-track data's original resolution
        # (such as ATCF given by the JTWC) is 6 hours. One point out of two was linearly interpolated
        # (which is very common with best-track data), which would make the problem artificially easier.
        # To avoid this, we'll only keep the points that are multiples of 6 hours.
        # Select the points in the info
        data_info = data_info[data_info["ISO_TIME"].dt.hour.isin([0, 6, 12, 18])]
        # Select the corresponding entries in the datacube
        datacube = datacube.isel(phony_dim_4=data_info.index)
        # Reset the index of data_info so that it matches datacube.isel
        data_info = data_info.reset_index(drop=True)

        # === DATACUBE PREPROCESSING ===
        # Rename the dimensions
        datacube = datacube.rename(
            {
                "phony_dim_4": "sid_time",
                "phony_dim_5": "v_pixel_offset",
                "phony_dim_6": "h_pixel_offset",
                "phony_dim_7": "variable",
            }
        )
        # Remove the Water vapor and Visible channels, as supposedly do not
        # provide information that complements the IR channel.
        # Besides, the Visible channel is unstable due to daylight.
        datacube = datacube.drop_sel(variable=[1, 2])

        # Add coordinates for the latitude and longitude dimensions
        # Since the lat/lon differs for each storm and time, we'll use the offset
        # (in pixels) from the center of the image
        img_side_len = datacube.shape[1]
        offset = np.arange(-(img_side_len // 2), img_side_len // 2 + 1)
        datacube = datacube.assign_coords(
            h_pixel_offset=("h_pixel_offset", offset), v_pixel_offset=("v_pixel_offset", offset)
        )

        # Add coordinates for the sid_time dimension
        # This coordinate will be in the format of a tuple (SID, ISO_TIME)
        datacube = datacube.assign_coords(
            sid=("sid_time", data_info["SID"].values),
            time=("sid_time", data_info["ISO_TIME"].values),
        )

        # Finally, we'll add a coordinate for the variable dimension
        datacube = datacube.assign_coords(variable=("variable", ["IR", "PMW"]))

        # === OUTLIERS AND MISSING VALUES ==== (SEE NOTEBOOK FOR EXPLAINATIONS)
        print("Processing outliers and missing values...")
        # Convert the unreasonably large values to NaNs
        datacube = datacube.where(datacube[:, :, :, 1] < 10 ^ 3)

        # Interpolate the partially-NaN images
        datacube = datacube.ffill("h_pixel_offset")
        datacube = datacube.ffill("v_pixel_offset")
        # NaN values at the border won't be interpolated, we'll fill them with zeros.
        datacube = datacube.fillna(0)

        # Save the intermediate datacube
        print("Saving the intermediate datacube and info...")
        datacube.to_netcdf(interm_cube_path, compute=True)
        # Save the intermediate info
        data_info.to_csv(interm_info_path, index=True)
    else:
        # Load the intermediate data
        print("Loading the intermediate data from ", save_dir)
        datacube = xr.open_dataarray(save_dir / "datacube_intermediate.nc")
        data_info = pd.read_csv(save_dir / "info_intermediate.csv", index_col=0)

    # === CONCATENATION WITH ERA5 =======
    # Load the ERA5 patches
    era5 = xr.open_mfdataset(
        _ERA5_PATCHES_PATH_ + "/*_surf*.nc",
        combine="nested",
        concat_dim="sid_time",
        chunks={"sid_time": "auto"},
    )
    # Add a 'sid_time' index to ERA5 and to TCIR
    era5 = era5.set_xindex(["sid", "time"])
    datacube = datacube.set_xindex(["sid", "time"])
    # Convert the TCIR DataArray to a Dataset so that we can use merge():
    datacube = datacube.to_dataset(dim="variable")
    # There might be a 1 shift between the spatial coords of the two datasets
    # (eg -101 to 99 for ERA5 and -100 to 100 for TCIR).
    # This was fixed in the preprocessing of the ERA5 patches, but I don't
    # have time to rerun it rn. So we'll just set the ERA5 coords to match
    # the TCIR coords :'). Anyway the resolution and patch size are the same.
    era5 = era5.assign_coords(
            h_pixel_offset=('h_pixel_offset', datacube.h_pixel_offset.data),
            v_pixel_offset=('v_pixel_offset', datacube.v_pixel_offset.data)
        )
    # Merge the ERA5 dataset with the TCIR dataset
    datacube = xr.merge([datacube, era5], compat="equals", join="left")
    # Destroy the MultiIndex, which won't be needed anymore and can't be saved to netCDF
    datacube = datacube.reset_index("sid_time")

    # === FEATURE ENGINEERING ===
    # Add the time of the day as a feature
    embedded_time = hours_to_sincos(data_info["ISO_TIME"])
    data_info["HOUR_SIN"] = embedded_time[:, 0]
    data_info["HOUR_COS"] = embedded_time[:, 1]
    # Add the month as a feature
    embedded_month = months_to_sincos(data_info["ISO_TIME"])
    data_info["MONTH_SIN"] = embedded_month[:, 0]
    data_info["MONTH_COS"] = embedded_month[:, 1]

    # === ADDITIONAL TARGETS ===
    # Add the SSHS category as a target
    # Add 1 so that the range is [0, 6] instead of [-1, 5]
    data_info["SSHS"] = sshs_category_array(data_info["INTENSITY"]) + 1

    # === SHIFTED FEATURES CONSTRUCTION ===
    # For all variables that will either be used as context or target, we'll add columns
    # giving their values shifted by a given set of time steps. We'll also add columns
    # giving the difference between the shifted values and the original values.
    # Example: INTENSITY_2 is the intensity 2 time steps in the future, and DELTA_INTENSITY_2
    # is the difference between INTENSITY_2 and INTENSITY.
    # 1. Load the set of time steps to shift the variables
    shifts = cfg["preprocessing"]["steps"]
    # 2. Fixed set of variables to shift and compute deltas
    shifted_vars = [
        "INTENSITY",
        "LON",
        "LAT",
        "R35_4qAVG",
        "MSLP",
        "SSHS",
        "HOUR_COS",
        "HOUR_SIN",
        "MONTH_COS",
        "MONTH_SIN",
    ]
    delta_cols = ["INTENSITY", "LON", "LAT", "R35_4qAVG", "MSLP"]
    # 3. Shift the variables and compute the deltas
    data_info = grouped_shifts_and_deltas(data_info, "SID", shifted_vars, delta_cols, shifts)
    # Remark: the shift operation creates NaNs in the first min(min(shifts), 0) and last
    # max(max(shifts), 0) rows, as their is no data to shift.
    # We need to keep those rows for the splitting
    # so that the index of the dataframe matches the index of the datacube.

    # === RAPID INTENSIFICATION LABEL ===
    # Rapid Intensification (RI) is defined by the NHC as an increase in MSW of at least 30 kt
    # in 24h. We'll add a binary label for RI, but at each time step. The RI label at t will
    # be 1 if Y_t - Y_0 >= (30 / 24) * t, where Y_t is the intensity at time t.
    # We'll use the DELTA_INTENSITY_t columns.
    for t in [s for s in shifts if s > 0]:
        data_info[f"RI_{t}"] = (data_info[f"DELTA_INTENSITY_{t}"] >= (30 / 24) * (t * 6)).astype(
            int
        )

    # === TRAIN/TEST SPLIT ===
    # Split into train/test sets
    train_idx, test_idx = stormwise_train_test_split(data_info, test_years=2016)
    # Re-split the training set into a training set and a validation set
    train_idx, val_idx = stormwise_train_test_split(
        data_info.loc[train_idx], test_years=[2014, 2015]
    )
    print("Train/Validation/Test split:")
    train_info = data_info.loc[train_idx].copy()
    val_info = data_info.loc[val_idx].copy()
    train_datacube = datacube.isel(sid_time=train_idx)
    test_info = data_info.loc[test_idx].copy()
    val_datacube = datacube.isel(sid_time=val_idx)
    test_datacube = datacube.isel(sid_time=test_idx)
    # Reset the index of the dataframes so that they match datacube.isel
    train_info = train_info.reset_index(drop=True)
    val_info = val_info.reset_index(drop=True)
    test_info = test_info.reset_index(drop=True)
    # We can now remove the rows which contain NaNs due to the shift operation
    train_info = train_info.dropna(axis="index")
    val_info = val_info.dropna(axis="index")
    test_info = test_info.dropna(axis="index")
    # Normalize the info and datacube using constants computed on the training set
    # Select a set of columns which won't be normalized (e.g. categorical columns)
    # Non-numeric columns will not be normalized
    # Only indicate the original name, e.g. "SSHS", not "SSHS_1", "SSHS_2", etc.
    non_normalized_cols = ["SSHS", "RI"]
    # Retrive the columns 'XXXX_t' from 'XXXX'
    non_normalized_cols = [
        col for col in train_info.columns if col.split("_")[0] in non_normalized_cols
    ]
    # Retrieve a copy of the numeric columns
    numeric_df = train_info.select_dtypes(include=[np.number]).copy()
    # Deduce the columns to normalize
    normalized_cols = [col for col in numeric_df.columns if col not in non_normalized_cols]
    numeric_df = numeric_df[normalized_cols]
    # First, normalize the whole training set and test set using the training set's mean and std
    # Info
    info_mean, info_std = numeric_df.mean(), numeric_df.std()
    # Datacube
    train_datacube_mean = train_datacube.mean(dim=["sid_time", "h_pixel_offset", "v_pixel_offset"])
    train_datacube_std = train_datacube.std(dim=["sid_time", "h_pixel_offset", "v_pixel_offset"])
    # Save the normalization constants
    print("Saving the normalization constants...")
    info_mean.to_csv(save_dir / "info_mean.csv", header=False)
    info_std.to_csv(save_dir / "info_std.csv", header=False)
    train_datacube_mean.to_netcdf(save_dir / "datacube_mean.nc")
    train_datacube_std.to_netcdf(save_dir / "datacube_std.nc")
    print("Normalizing the data...")
    # Perform the normalization on the info
    train_info[normalized_cols] = (train_info[normalized_cols] - info_mean) / info_std
    val_info[normalized_cols] = (val_info[normalized_cols] - info_mean) / info_std
    test_info[normalized_cols] = (test_info[normalized_cols] - info_mean) / info_std
    # Perform the normalization on the datacube
    # But first, reload the means and stds from the files
    # (see https://docs.xarray.dev/en/stable/user-guide/dask.html#chunking-and-performance)
    train_datacube_mean = xr.open_dataset(save_dir / "datacube_mean.nc")
    train_datacube_std = xr.open_dataset(save_dir / "datacube_std.nc")
    train_datacube = (train_datacube - train_datacube_mean) / train_datacube_std
    val_datacube = (val_datacube - train_datacube_mean) / train_datacube_std
    test_datacube = (test_datacube - train_datacube_mean) / train_datacube_std

    print("Computing with Dask and saving...")
    # Retrieve the path to the save directory
    save_dir = Path(cfg["paths"]["tcir_preprocessed_dir"])
    # Create train/test subdirectories if they don't exist
    for subdir in ["train", "val", "test"]:
        subdir_path = save_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
    # Save the data
    with ProgressBar():
        print("Training datacube...")
        train_datacube.to_netcdf(save_dir / "train" / "datacube.nc")
    with ProgressBar():
        print("Validation datacube...")
        val_datacube.to_netcdf(save_dir / "val" / "datacube.nc")
    with ProgressBar():
        print("Test datacube...")
        test_datacube.to_netcdf(save_dir / "test" / "datacube.nc")
    print("Saving the tabular data...")
    train_info.to_csv(save_dir / "train" / "info.csv", index=True)
    val_info.to_csv(save_dir / "val" / "info.csv", index=True)
    test_info.to_csv(save_dir / "test" / "info.csv", index=True)
