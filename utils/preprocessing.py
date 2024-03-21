def grouped_shifts_and_deltas(df, group_column, target_columns, steps):
    """
    Given a dataframe (Vg, V1, V2, ..., VK, ...) where Vg is a grouping column,
    V1, V2, ..., VK are target columns, and a list of integer steps,
    * Computes the targets columns shifted by each step within each group, and
      adds them as columns "V1_{steps[0]}", "V1_{steps[1]}", ..., "VK_{steps[-1]}".
      For example, a step of 1 means the next row, a step of -1 means the previous row.
    * Computes the deltas between the shifted columns and the original columns, and
      adds them as columns "DELTA_V1_{steps[0]}", "DELTA_V1_{steps[1]}", ..., "DELTA_VK_{steps[-1]}".
      Note: DELTA_Vi_j = Vi_j - Vi (i.e. Vi_0), not Vi_j - Vi_{j-1}.
    The first min(min(step), 0) rows and the last (max(max(step), 0) rows of each group are removed,
    as they would contain NaN due to the shift. The index of the kept rows is NOT modified.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to compute deltas for.
    group_column : str
        The name of the column to group by.
    target_columns : list of str
        The names of the columns to compute deltas for.

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
    for target_column in target_columns:
        # For each step
        for step in steps:
            step = int(step)
            # Compute the shifted column
            new_df[f"{target_column}_{step}"] = grouped[target_column].shift(-step)
            # Compute the delta column
            new_df[f"DELTA_{target_column}_{step}"] = (
                new_df[f"{target_column}_{step}"] - new_df[target_column]
            )
    # Remove the first min(min(step), 0) rows and the last max(max(step), 0) rows
    # of each group.
    new_df = new_df.groupby(group_column).apply(
        lambda g: g.iloc[-min(min(steps), 0) : -max(max(steps), 0)]
    )
    # Don't reset the index to keep the original one for the remaining rows
    # Instead, we can retrieve the index of the remaining rows as the second level
    # of the multi-index created by groupby
    new_df.set_index(new_df.index.get_level_values(1), inplace=True)
    new_df = new_df.sort_index()

    return new_df
