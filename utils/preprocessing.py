def grouped_shifts_and_deltas(df, group_column, shifted_columns, delta_columns, steps):
    """
    Given a dataframe (Vg, V1, V2, ..., VK, ...) where Vg is a grouping column,
    V1, V2, ..., VK are target columns, and a list of integer steps,
    * Computes the targets columns shifted by each step within each group, and
      adds them as columns "V1_{steps[0]}", "V1_{steps[1]}", ..., "VK_{steps[-1]}".
      For example, a step of 1 means the next row, a step of -1 means the previous row.
    * Computes the deltas between the shifted columns and the original columns, and
      adds them as columns "DELTA_V1_{steps[0]}", "DELTA_V1_{steps[1]}", ..., "DELTA_VK_{steps[-1]}".
      Note: DELTA_Vi_j = Vi_j - Vi (i.e. Vi_0), not Vi_j - Vi_{j-1}.
    Remark: The first min(min(steps), 0) rows and the last max(max(steps), 0) rows of each group
    contain NaNs due to the shift operation.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to compute deltas for.
    group_column : str
        The name of the column to group by.
    shifted_columns : list of str
        The names of the columns to compute deltas for.
    delta_columns: list of str
        The names of the columns to compute deltas for. Must be a subset of shifted_columns.

    Returns
    -------
    pd.DataFrame
        The dataframe with the shifted and delta columns added.
    """
    # Initialize the new dataframe
    new_df = df.copy()

    # Group by the group column
    grouped = new_df.groupby(group_column)

    # For each target column
    for shifted_column in shifted_columns:
        # For each step
        for step in steps:
            step = int(step)
            # Compute the shifted column
            new_df[f"{shifted_column}_{step}"] = grouped[shifted_column].shift(-step)
            # Compute the delta column
            if shifted_column in delta_columns:
                new_df[f"DELTA_{shifted_column}_{step}"] = (
                    new_df[f"{shifted_column}_{step}"] - new_df[shifted_column]
                )

    return new_df
