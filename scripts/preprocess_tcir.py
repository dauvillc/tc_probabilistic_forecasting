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
from tqdm import tqdm
from utils.utils import hours_to_sincos
from utils.preprocessing import grouped_shifts_and_deltas
from utils.train_test_split import stormwise_train_test_split, kfold_split


if __name__ == "__main__":
    # Set numpy's random seed for reproducibility
    np.random.seed(0)
    # Load config file
    with open("config.yml", "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    _TCIR_PATH_ = cfg["paths"]["tcir"]

    print("Loading the raw TCIR dataset...")
    # Load the tabular information
    data_info = pd.read_hdf(_TCIR_PATH_, key="info", mode="r")

    # Load the datacube
    datacube = xr.open_dataset(_TCIR_PATH_)["matrix"]

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
        sids = np.random.choice(sids, size=int(len(sids) * fraction), replace=False)
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
        sid=("sid_time", data_info["SID"].values), time=("sid_time", data_info["ISO_TIME"].values)
    )

    # Finally, we'll add a coordinate for the variable dimension
    datacube = datacube.assign_coords(variable=("variable", ["IR", "PMW"]))

    # === OUTLIERS AND MISSING VALUES ==== (SEE NOTEBOOK FOR EXPLAINATIONS)
    print("Processing outliers and missing values...")
    # Convert the unreasonably large values to NaNs
    datacube = datacube.where(datacube[:, :, :, 1] < 10 ^ 3)
    # Compute the ratio of NaNs (native + converted from outliers) for each sample
    nan_counts = np.array([datacube[k].isnull().sum() for k in range(datacube.shape[0])])
    nan_ratios = nan_counts / (datacube.shape[1] * datacube.shape[2] * 2)
    print("Average NaN ratio: ", nan_ratios.mean(), " std: ", nan_ratios.std())

    # * For every storm S that contains at least one sample that is more than 10% NaN:
    #   * Find all segments of S's trajectory that do not contain any NaN;
    #   * Consider each of those segments as an independent trajectory, by giving them new SIDs.
    #   * Discard the samples containing more than 10% of NaNs.
    # * Where the images contain less than 10% of NaNs, fill them with zeros.

    # Retrieve the index of the samples that are fully NaN (in the sense of over 10%)
    where_full_nan = np.where(nan_ratios >= 0.1)[0]
    # Retrieve the SIDs corresponding to those samples
    sids_with_full_nan = data_info.iloc[where_full_nan]["SID"].unique()
    print('"Full" NaN proportion: ', where_full_nan.shape[0] / datacube.shape[0])
    print(
        'Proportion of storms that include at least one "full" NaN sample: ',
        sids_with_full_nan.shape[0] / data_info["SID"].unique().shape[0],
    )

    print("Processing storms including at least one full-NaN sample...")
    for sid in tqdm(sids_with_full_nan):
        segments = []
        current_segment = []
        currently_in_segment = False
        # Iterate through the samples of that storm. If we find a NaN, that's the end
        # of the current segment. Keep iterating until finding a non full-NaN sample,
        # which is the start of a new segment.
        for k in data_info[data_info["SID"] == sid].index:
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
        # Add the last segment
        if currently_in_segment:
            segments.append(np.array(current_segment))
        # Give all samples in that segment a new SID
        sid_values = data_info["SID"].values.copy()
        for n_seg, seg in enumerate(segments):
            sid_values[seg] = sid_values[seg] + f"_{n_seg}"
        data_info["SID"] = sid_values
    # Drop the samples that are fully NaNs
    data_info = data_info.drop(index=where_full_nan)
    # Select the index of data_info (which doesn't contain the full-NaN samples anymore) in the datacube
    datacube = datacube.isel(sid_time=data_info.index)
    # Reset the index of the tabular data, so that it matches that of the datacube
    data_info = data_info.reset_index(drop=True)
    # We can now interpolate the partially-NaN images
    datacube = datacube.interpolate_na("h_pixel_offset", method="nearest")
    # NaN values at the border won't be interpolated, we'll fill them with zeros.
    datacube = datacube.fillna(0)

    # === FEATURE ENGINEERING ===
    # Add the time of the day as a feature
    embedded_time = hours_to_sincos(data_info["ISO_TIME"])
    data_info["HOUR_SIN"] = embedded_time[:, 0]
    data_info["HOUR_COS"] = embedded_time[:, 1]

    # === SHIFTED FEATURES CONSTRUCTION ===
    # For all variables that will either be used as context or target, we'll add columns
    # giving their values shifted by a given set of time steps. We'll also add columns
    # giving the difference between the shifted values and the original values.
    # Example: INTENSITY_2 is the intensity 2 time steps in the future, and DELTA_INTENSITY_2
    # is the difference between INTENSITY_2 and INTENSITY.
    # 1. Load the set of time steps to shift the variables
    shifts = cfg["preprocessing"]["steps"]
    # 2. Fixed set of variables to shift and compute deltas
    shifted_vars = ["INTENSITY", "LON", "LAT", "R35_4qAVG", "MSLP", "HOUR_COS", "HOUR_SIN"]
    delta_cols = ["INTENSITY", "LON", "LAT", "R35_4qAVG", "MSLP"]
    # 3. Shift the variables and compute the deltas
    data_info = grouped_shifts_and_deltas(data_info, "SID", shifted_vars, delta_cols, shifts)
    # Remark: the shift operation creates NaNs in the first min(min(shifts), 0) and last
    # max(max(shifts), 0) rows, as their is no data to shift.
    # We need to keep those rows for the splitting
    # so that the index of the dataframe matches the index of the datacube.

    # === TRAIN/VAL/TEST SPLIT ===
    # Split into train/test sets
    train_idx, test_idx = stormwise_train_test_split(data_info, train_size=0.8, test_size=0.2)
    print("Train/test split:", f"{len(train_idx)} / {len(test_idx)}")
    print("Splitting the training set into K folds...")
    train_info = data_info.loc[train_idx].copy()
    train_datacube = datacube.isel(sid_time=train_idx)
    test_info = data_info.loc[test_idx].copy()
    test_datacube = datacube.isel(sid_time=test_idx)
    # Split the training set into K folds
    splits = kfold_split(data_info, n_splits=5)
    # Deduce the training and validation sets from the splits
    splits_info = [
        (data_info.loc[train_idx].copy(), data_info.loc[val_idx].copy())
        for train_idx, val_idx in splits
    ]
    splits_datacube = [
        (datacube.isel(sid_time=train_idx), datacube.isel(sid_time=val_idx))
        for train_idx, val_idx in splits
    ]
    # Reset the index of the dataframes so that they match datacube.isel
    train_info = train_info.reset_index(drop=True)
    test_info = test_info.reset_index(drop=True)
    for k, (sp_train_info, sp_val_info) in enumerate(splits_info):
        splits_info[k] = (sp_train_info.reset_index(drop=True), sp_val_info.reset_index(drop=True))
    # We can now remove the rows which contain NaNs due to the shift operation
    train_info = train_info.dropna(axis='index')
    test_info = test_info.dropna(axis='index')
    for k, (sp_train_info, sp_val_info) in enumerate(splits_info):
        splits_info[k] = (sp_train_info.dropna(axis='index'), sp_val_info.dropna(axis='index'))

    # === NORMALIZATION ===
    # Normalize the info and datacube using constants computed on the training set
    print("Normalizing ...")
    # Only normalize the numeric columns
    numeric_df = train_info.select_dtypes(include=[np.number]).copy()
    numeric_cols = numeric_df.columns
    # First, normalize the whole training set and test set using the training set's mean and std
    # Info
    info_mean, info_std = numeric_df.mean(), numeric_df.std()
    train_info[numeric_cols] = (train_info[numeric_cols] - info_mean) / info_std
    test_info[numeric_cols] = (test_info[numeric_cols] - info_mean) / info_std
    # Datacube
    train_datacube_mean = train_datacube.mean(dim=["sid_time", "h_pixel_offset", "v_pixel_offset"])
    train_datacube_std = train_datacube.std(dim=["sid_time", "h_pixel_offset", "v_pixel_offset"])
    train_datacube = (train_datacube - train_datacube_mean) / train_datacube_std
    test_datacube = (test_datacube - train_datacube_mean) / train_datacube_std
    # Now, for each fold, normalize the training and validation sets using the training set's mean and std
    # Store the normalization constants
    sp_info_means, sp_info_stds = [], []
    sp_datacube_means, sp_datacube_stds = [], []
    for k, ((sp_train_info, sp_val_info), (sp_train_datacube, sp_val_datacube)) in enumerate(
        zip(splits_info, splits_datacube)
    ):
        # Info
        sp_info_mean, sp_info_std = sp_train_info[numeric_cols].mean(), sp_train_info[numeric_cols].std()
        sp_train_info[numeric_cols] = (sp_train_info[numeric_cols] - sp_info_mean) / sp_info_std
        sp_val_info[numeric_cols] = (sp_val_info[numeric_cols] - sp_info_mean) / sp_info_std
        # Datacube
        sp_datacube_mean = sp_train_datacube.mean(
            dim=["sid_time", "h_pixel_offset", "v_pixel_offset"]
        )
        sp_datacube_std = sp_train_datacube.std(dim=["sid_time", "h_pixel_offset", "v_pixel_offset"])
        sp_train_datacube = (sp_train_datacube - sp_datacube_mean) / sp_datacube_std
        sp_val_datacube = (sp_val_datacube - sp_datacube_mean) / sp_datacube_std
        # Update the splits
        splits_info[k] = (sp_train_info, sp_val_info)
        splits_datacube[k] = (sp_train_datacube, sp_val_datacube)
        # Store the normalization constants
        sp_info_means.append(sp_info_mean)
        sp_info_stds.append(sp_info_std)
        sp_datacube_means.append(sp_datacube_mean)
        sp_datacube_stds.append(sp_datacube_std)

    # === SAVE THE PREPROCESSED DATA ===
    print("Saving the preprocessed data...")
    # Retrieve the path to the save directory
    save_dir = Path(cfg["paths"]["tcir_preprocessed_dir"])
    # Create train/val/test subdirectories if they don't exist
    for subdir in ["train", "test"]:
        subdir_path = save_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
    # Save the whole training set and the test set
    train_datacube.to_netcdf(save_dir / "train" / "datacube.nc")
    test_datacube.to_netcdf(save_dir / "test" / "datacube.nc")
    train_info.to_csv(save_dir / "train" / "info.csv", index=True)
    test_info.to_csv(save_dir / "test" / "info.csv", index=True)
    # Save the normalization constants
    info_mean.to_csv(save_dir / "info_mean.csv", header=False)
    info_std.to_csv(save_dir / "info_std.csv", header=False)
    train_datacube_mean.to_netcdf(save_dir / "datacube_mean.nc")
    train_datacube_std.to_netcdf(save_dir / "datacube_std.nc")
    # Save the splits
    for k, ((sp_train_info, sp_val_info), (sp_train_datacube, sp_val_datacube)) in enumerate(
        zip(splits_info, splits_datacube)
    ):
        # Create the subdirectories
        for subdir in ["train", "val"]:
            subdir_path = save_dir / f"split_{k}" / subdir
            subdir_path.mkdir(parents=True, exist_ok=True)
        # Save the split
        sp_train_datacube.to_netcdf(save_dir / f"split_{k}" / "train" / "datacube.nc")
        sp_val_datacube.to_netcdf(save_dir / f"split_{k}" / "val" / "datacube.nc")
        sp_train_info.to_csv(save_dir / f"split_{k}" / "train" / "info.csv", index=True)
        sp_val_info.to_csv(save_dir / f"split_{k}" / "val" / "info.csv", index=True)
        # Save the normalization constants
        sp_info_means[k].to_csv(save_dir / f"split_{k}" / "info_mean.csv", header=False)
        sp_info_stds[k].to_csv(save_dir / f"split_{k}" / "info_std.csv", header=False)
        sp_datacube_means[k].to_netcdf(save_dir / f"split_{k}" / "datacube_mean.nc")
        sp_datacube_stds[k].to_netcdf(save_dir / f"split_{k}" / "datacube_std.nc")
