"""
Clément Dauvilliers - 2023/10/17
Implements functions to help with train / val / test splitting.
"""

import numpy as np
from sklearn.model_selection import train_test_split, KFold


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
    merged_sids = np.array(list(set([sid.split('_')[0] for sid in sids])))
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


def stormwise_train_test_split(ibtracs_data, train_size=0.8, test_size=0.2, random_state=42):
    """
    Splits the IBTrACS dataset into train and test sets.
    The returned splits contain different storms and thus disjoint tracks.
    If a trajectory "XXXX" is split into subtrajectories "XXXX_0", "XXXX_1", etc.,
    then all of these subtrajectories are included in the same split.

    Parameters
    ----------
    ibtracs_data : pandas.DataFrame
        Preprocessed [subset of the] IBTrACS dataset.
        The dataset must be ordered by SID, and then valid time.
        The SIDs must be ordered by time of the first record of the storm (as in the original dataset).
    train_size : float, optional.
        Size of the train set.
    test_size : float, optional.
        Size of the test set.
    random_state : int, optional.
        If None, the random state is still set to a default value for reproducibility.

    Returns
    -------
    train_indices : numpy.ndarray
        Indices of the train set in the IBTrACS dataset.
    test_indices : numpy.ndarray
        Indices of the test set in the IBTrACS dataset.
    """
    # Retrieve all unique SIDs
    sids = ibtracs_data['SID'].unique()
    # Some SIDs are "XXXX_k" to indicate the kth subtrajectory of the storm "XXXX".
    # We want to keep these subtrajectories together in the same split.
    # We'll thus ignore the "_k" suffix while splitting the dataset.
    merged_sids = list(set([sid.split('_')[0] for sid in sids]))

    # Split the SIDs into train, val and test sets
    train_sids, test_sids = train_test_split(merged_sids,
                                             train_size=train_size,
                                             test_size=test_size,
                                             random_state=random_state)
    # Add back the "_k" suffix to the SIDs: if 'XXXX' is in a split, add all 'XXXX_k' that are
    # in sids to the split as well.
    train_sids = [sid for sid in sids if sid.split('_')[0] in train_sids]
    test_sids = [sid for sid in sids if sid.split('_')[0] in test_sids]
    # Retrieve the indices of the train, val and test sets from the SIDs.
    train_indices = ibtracs_data[ibtracs_data['SID'].isin(train_sids)].index.values
    test_indices = ibtracs_data[ibtracs_data['SID'].isin(test_sids)].index.values

    return train_indices, test_indices

