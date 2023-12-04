"""
Clément Dauvilliers - 2023/10/17
Implements functions to help with train / val / test splitting.
"""

from sklearn.model_selection import train_test_split


def train_val_test_split(ibtracs_data, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """
    Splits the IBTrACS dataset into train, validation and test sets.
    The returned splits contain different storms and thus disjoint tracks.

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

    # Split the SIDs into train, val and test sets
    train_sids, test_sids = train_test_split(sids,
                                             train_size=(train_size + val_size),
                                             test_size=test_size,
                                             random_state=random_state)
    train_sids, val_sids = train_test_split(train_sids,
                                            train_size=train_size / (train_size + val_size),
                                            test_size=val_size / (train_size + val_size),
                                            random_state=random_state)
    # Retrieve the indices of the train, val and test sets from the SIDs.
    train_indices = ibtracs_data[ibtracs_data['SID'].isin(train_sids)].index.values
    val_indices = ibtracs_data[ibtracs_data['SID'].isin(val_sids)].index.values
    test_indices = ibtracs_data[ibtracs_data['SID'].isin(test_sids)].index.values

    return train_indices, val_indices, test_indices

