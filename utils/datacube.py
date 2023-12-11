"""
Clément Dauvilliers - 2023 10 18
Implements function to manipulate datacubes.
"""
import torch
import xarray as xr
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


def rescale_and_crop(datacube, dims, new_res=None, crop_size=None):
    """
    Spatially rescales a datacube (either up or down) and crops its center.
    The interpolation is linear.

    Parameters
    ----------
    datacube : xarray.Dataset
        Datacube to be upscaled.
    dims : tuple of str
        Names of the latitude and longitude dimensions, which shoudl be in the same
        unit as new_res.
        Both dimensions must have the same size.
    new_res : float, optional
        New resolution of the datacube. Default is None, which means no rescaling.
    crop_size : int
        Size of the crop, in pixels. Default is None, which means no crop.

    Returns
    -------
    datacube : xarray.Dataset
        Upscaled datacube.
    """
    if datacube[dims[0]].shape != datacube[dims[1]].shape:
        raise ValueError("The dimensions to be upscaled must have the same size.")
    if new_res is not None:
        # Retrieve the min and max values of the dimensions to be upscaled
        min_lat, max_lat = datacube[dims[0]].min(), datacube[dims[0]].max()
        min_lon, max_lon = datacube[dims[1]].min(), datacube[dims[1]].max()
        # Create the new coordinates for latitude and longitude, with points spaced by the new resolution
        new_lat = np.arange(min_lat, max_lat + new_res, new_res)
        new_lon = np.arange(min_lon, max_lon + new_res, new_res)
        # Crop the center of the coordinates if needed
        if crop_size is not None:
            center_idx = len(new_lat) // 2
            new_lat = new_lat[center_idx - crop_size // 2: center_idx + crop_size // 2 + crop_size % 2]
            new_lon = new_lon[center_idx - crop_size // 2: center_idx + crop_size // 2 + crop_size % 2]

        # Rescale the datacube
        datacube = datacube.interp({dims[0]: new_lat, dims[1]: new_lon}, method='linear')
        return datacube
    else:
        # If no rescaling is needed, just crop the center of the datacube
        if crop_size is not None:
            center_idx = len(datacube[dims[0]]) // 2
            new_idx = slice(center_idx - crop_size // 2, center_idx + crop_size // 2 + crop_size % 2)
            datacube = datacube.isel({dims[0]: new_idx, dims[1]: new_idx})
        return datacube


def datacube_to_tensor(datacube, dim_h='h_pixel_offset', dim_v='v_pixel_offset'):
    """
    Converts a datacube stored as an xarray Dataset or DataArray
    to a torch tensor.

    Parameters
    ----------
    datacube : xarray Dataset or DataArray of dimensions (sid_time, dim_h, dim_v [, ...]).
    dim_h : str, optional
        Name of the horizontal (i.e. latitude-like) dimension. 
    dim_v : str, optional
        Name of the vertical (i.e. longitude-like) dimension.

    Returns
    -------
    torch tensor of dimensions (sid_time, C, H, W) where C is the number of channels.
    """
    # If the datacube is a Dataset, convert it to a DataArray by stacking all variables
    # in one dimension
    if isinstance(datacube, xr.Dataset):
        datacube = datacube.to_array(dim="variable")
    # Stack all dimensions except sid_time, dim_h and dim_v to obtain a DataArray of
    # dimensions (sid_time, channels, dim_h, dim_v)
    other_dims = [dim for dim in datacube.dims if dim not in ['sid_time', dim_h, dim_v]]
    if other_dims:
        datacube = datacube.stack(channels=other_dims)
    else:
        # If the datacube is only 3D, add a dummy dimension to make it 4D
        datacube = datacube.expand_dims('channels', axis=1)
    # Reorder the dimensions to (N, C, H, W) with N = sid_time,
    # C = channels, H = dim_h, W = dim_v to be compatible with the input of a torch CNN
    datacube = datacube.transpose("sid_time", "channels", dim_h, dim_v)
    # Convert to a torch tensor
    datacube = torch.tensor(datacube.values, dtype=torch.float32)
    return datacube
