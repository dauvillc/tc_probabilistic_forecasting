"""
Clément Dauvilliers - 2023 10 18
Implements function to manipulate datacubes.
"""
import numpy as np


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


def set_relative_spatial_coords(datacube, lat_dim="latitude", lon_dim="longitude"):
    """
    Given a datacube indexed by absolute latitude and longitude values, returns a
    datacube indexed by relative latitude and longitude offsets from the storm's center.

    Parameters
    ----------
    datacube : xarray.DataArray
        Datacube to be manipulated.
    lat_dim : str, optional
        Name of the latitude dimension. Default is "latitude".
    lon_dim : str, optional
        Name of the longitude dimension. Default is "longitude".

    Returns
    -------
    datacube : xarray.Dataset
        Datacube indexed by relative latitude and longitude offsets from the storm's center.
        The new dimensions are "h_pixel_offset" and "v_pixel_offset".
    """
    v_size, h_size = datacube[lat_dim].shape[0], datacube[lon_dim].shape[0]
    # Rename the latitude and longitude dimensions, and drop the former coordinates.
    datacube = datacube.rename({lon_dim: 'h_pixel_offset', lat_dim: 'v_pixel_offset'})
    # Create the new coordinates, one for the vertical offset and one for the horizontal offset
    vertical_coords = np.arange(-v_size // 2, v_size // 2)
    horizontal_coords = np.arange(-h_size // 2, h_size // 2)
    # Assign the new coordinates to the datacube
    datacube = datacube.assign_coords(v_pixel_offset=('v_pixel_offset', vertical_coords),
                                      h_pixel_offset=('h_pixel_offset', horizontal_coords))
    return datacube
