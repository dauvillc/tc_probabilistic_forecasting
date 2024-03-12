"""
Clément Dauvilliers - 2023/10/17
Implements functions to help with train / val / test splitting.
"""

from sklearn.model_selection import train_test_split


def train_val_test_split(ibtracs_data, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """
    Splits the IBTrACS dataset into train, validation and test sets.
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
        Size of the train set. The default is 0.6.
    val_size : float, optional.
        Size of the validation set. The default is 0.2.
    test_size : float, optional.
        Size of the test set. The default is 0.2.
    random_state : int, optional.
        If None, the random state is still set to a default value for reproducibility.

    Returns
    -------
    train_indices : numpy.ndarray
        Indices of the train set in the IBTrACS dataset.
    val_indices : numpy.ndarray
        Indices of the validation set in the IBTrACS dataset.
    test_indices : numpy.ndarray
        Indices of the test set in the IBTrACS dataset.
    """
    # Retrieve all unique SIDs
    sids = ibtracs_data['SID'].unique()
    # Some SIDs are "XXXX_k" to indicate the kth subtrajectory of the storm "XXXX".
    # We want to keep these subtrajectories together in the same split.
    # We'll thus ignore the "_k" suffix while splitting the dataset.
    merged_sids = [sid.split('_')[0] for sid in sids]

    # Split the SIDs into train, val and test sets
    train_sids, test_sids = train_test_split(merged_sids,
                                             train_size=(train_size + val_size),
                                             test_size=test_size,
                                             random_state=random_state)
    train_sids, val_sids = train_test_split(train_sids,
                                            train_size=train_size / (train_size + val_size),
                                            test_size=val_size / (train_size + val_size),
                                            random_state=random_state)
    # Add back the "_k" suffix to the SIDs: if 'XXXX' is in a split, add all 'XXXX_k' that are
    # in sids to the split as well.
    train_sids = [sid for sid in sids if sid.split('_')[0] in train_sids]
    val_sids = [sid for sid in sids if sid.split('_')[0] in val_sids]
    test_sids = [sid for sid in sids if sid.split('_')[0] in test_sids]
    # Retrieve the indices of the train, val and test sets from the SIDs.
    train_indices = ibtracs_data[ibtracs_data['SID'].isin(train_sids)].index.values
    val_indices = ibtracs_data[ibtracs_data['SID'].isin(val_sids)].index.values
    test_indices = ibtracs_data[ibtracs_data['SID'].isin(test_sids)].index.values

    return train_indices, val_indices, test_indices

