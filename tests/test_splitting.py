"""
Clément Dauvilliers - 2023 10 17
Test the splitting of the data
"""

import sys
sys.path.append("./")
from utils.train_test_split import train_val_test_split
from data_processing import load_ibtracs_data


def count_storms(ibtracs_subset):
    """
    Count the number of storms in the given subset of IBTrACS data
    """
    return ibtracs_subset['SID'].nunique()


if __name__ == "__main__":
    # Load the preprocessed IBTrACS data
    ibtracs_data = load_ibtracs_data()

    # Split the storms and retrieve the indices
    train_indices, val_indices, test_indices = train_val_test_split(ibtracs_data)

    # Print the number of storms and points in each set
    print("Number of storms in the training set:", count_storms(ibtracs_data.iloc[train_indices]))
    print("Number of storms in the validation set:", count_storms(ibtracs_data.iloc[val_indices]))
    print("Number of storms in the test set:", count_storms(ibtracs_data.iloc[test_indices]))
    print("Number of points in the training set:", len(train_indices))
    print("Number of points in the validation set:", len(val_indices))
    print("Number of points in the test set:", len(test_indices))

