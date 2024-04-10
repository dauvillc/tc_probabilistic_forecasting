"""
Clément Dauvilliers - 2023/10/17
Implements functions to help with train / val / test splitting.
"""

import numpy as np
from sklearn.model_selection import KFold


def kfold_split(trajectories, n_splits=5, random_state=42):
    """
    Splits the trajectories into train and validation sets using K-Fold cross-validation.
    Samples from the same storm are kept together in the same fold.

    Parameters
    ----------
    trajectories : pandas.DataFrame
        Trajectories dataset including at least the column 'SID'.
    n_splits : int, optional
        Number of splits. The default is 5.
    random_state : int, optional
        Random seed.
    
    Returns
    -------
    splits: list of tuples
        Returns K tuples (train_indices, val_indices).
    """
    # Retrieve all unique SIDs
    sids = trajectories['SID'].unique()
    # Some SIDs are "XXXX_k" to indicate the kth subtrajectory of the storm "XXXX".
    # We want to keep these subtrajectories together in the same split.
    # We'll thus ignore the "_k" suffix while splitting the dataset.
    merged_sids = np.unique(np.array([sid.split('_')[0] for sid in sids]))
    # Perform K-Fold cross-validation on the merged SIDs
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    splits = []
    for train_index, val_index in kf.split(merged_sids):
        # Retrieve all SIDs of the form 'XXXX_j' where 'XXXX' is in the train or val index
        train_sids = [sid for sid in sids if sid.split('_')[0] in merged_sids[train_index]]
        val_sids = [sid for sid in sids if sid.split('_')[0] in merged_sids[val_index]]
        # Split the samples depending on where their SIDs fell
        train_indices = trajectories[trajectories['SID'].isin(train_sids)].index.values
        val_indices = trajectories[trajectories['SID'].isin(val_sids)].index.values
        splits.append((train_indices, val_indices))
    return splits


def stormwise_train_test_split(ibtracs_data, train_years=None, test_years=2017, random_state=42):
    """
    Splits the IBTrACS dataset into train and test sets.

    Parameters
    ----------
    ibtracs_data : pandas.DataFrame
        Preprocessed [subset of the] IBTrACS dataset.
        The dataset must be ordered by SID, and then valid time.
        The SIDs must be ordered by time of the first record of the storm (as in the original dataset).
    train_years : int or list of int
        Years to include in the train set. Defaults to all years except the test years.
    test_years : int or list of int
        Years to include in the test set. Defaults to 2017.
    random_state : int, optional.
        If None, the random state is still set to a default value for reproducibility.

    Returns
    -------
    train_indices : numpy.ndarray
        Indices of the train set in the IBTrACS dataset.
    test_indices : numpy.ndarray
        Indices of the test set in the IBTrACS dataset.
    """
    # Retrieve the indices of the train, val and test sets from the SIDs.
    # A SID has the form 'YEARXXX...'. We can thus extract the year from the first 4 characters.
    years = np.array([int(sid[:4]) for sid in ibtracs_data['SID']])
    # Retrieve the years to include in the train and test sets
    if isinstance(test_years, int):
        test_years = [test_years]
    if train_years is None:
        train_years = [year for year in np.unique(years) if year not in test_years]
    elif isinstance(train_years, int):
        train_years = [train_years]
    elif isinstance(train_years, list):
        # Check that the train and test years are disjoint
        if any(year in train_years for year in test_years):
            raise ValueError("The train and test years must be disjoint.")
        train_years = np.array(train_years)
    train_indices = ibtracs_data[np.isin(years, train_years)].index.values
    test_indices = ibtracs_data[np.isin(years, test_years)].index.values

    return train_indices, test_indices

