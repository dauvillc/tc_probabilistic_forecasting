"""
Clément Dauvilliers - 2023 10 18
Implements function to manipulate datacubes.
"""


def select_sid_time(datacube, sids, times):
    """
    Selects specific values of SID and time in a datacube.
    
    Parameters
    ----------
    datacube : xarray.Dataset
        Datacube to be manipulated, including dimension 'sid_time'. The 'sid_time'
        dimension must have the associated coordinates 'sid' and 'time'.
    sids : array-like
        List of SID values to be selected.
    times : array-like
        List of time values to be selected.

    Returns
    -------
    datacube : xarray.Dataset
        Datacube with the selected values of SID and time.
    """
    # Check if the datacube has the 'sid_time' dimension
    if 'sid_time' not in datacube.dims:
        raise ValueError("The datacube must have the 'sid_time' dimension.")
    # Check if the 'sid_time' dimension has the associated coordinates 'sid' and 'time'
    if 'sid' not in datacube.coords or 'time' not in datacube.coords:
        raise ValueError("The 'sid_time' dimension must have the associated coordinates 'sid' and 'time'.")
    # Check if the 'sid_time' MultiIndex already exists
    if 'sid_time' not in datacube.indexes:
        # Create the 'sid_time' MultiIndex
        datacube = datacube.set_index(sid_time=['sid', 'time'])
    # Select the values of SID and time
    # For that, we need to create a list of tuples (sid, time)
    sid_time = [(sid, time) for sid, time in zip(sids, times)]
    # Select the values of SID and time
    datacube = datacube.sel(sid_time=sid_time)

    return datacube
