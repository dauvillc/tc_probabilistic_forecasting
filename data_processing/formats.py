"""
Clément Dauvilliers - 2023 10 18
Implements functions to convert datacubes between different formats.
"""
import torch


def era5_patches_to_tensors(datacube):
    """
    Converts a datacube of ERA5 patches stored as an xarray Dataset
    to a torch tensor.

    Parameters
    ----------
    datacube : xarray Dataset of dimensions (sid_time, v_pixel_offset,
        h_pixel_offset [, levels]) with V variables.

    Returns
    -------
    torch tensor of dimensions (sid_time, V or V * levels],
        v_pixel_offset, h_pixel_offset)
    """
    # Convert to a DataArray by stacking all variables in one dimension
    datacube = datacube.to_array(dim="variable")
    # Check if "level" is in the dimensions. If so, stack the level and
    # the variable dimensions to obtain a DataArray of dimensions
    # (sid_time, level * variable, v_pixel_offset, h_pixel_offset)
    if "level" in datacube.dims:
        datacube = datacube.stack(channels=("level", "variable"))
    else:
        # Rename the "variable" dimension to "channels" to be compatible
        datacube = datacube.rename(variable="channels")
    # Reorder the dimensions to (N, C, H, W) with N = sid_time,
    # C = variable * level, H = v_pixel_offset, W = h_pixel_offset
    # to be compatible with the input of a torch CNN
    datacube = datacube.transpose("sid_time", "channels", ...)
    # Convert to a torch tensor
    datacube = torch.tensor(datacube.values, dtype=torch.float32)

    return datacube
