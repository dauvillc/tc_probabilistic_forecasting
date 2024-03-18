"""
Implements the SpatialEncoder class.
"""

import torch.nn as nn
from models.cnn3d import DownsamplingBlock3D


class SpatialEncoder(nn.Module):
    """
    The spatial encoder separately encodes the input images of a time series into lower-dimensional
    representations.
    In order to apply the encoder to multiple time steps at once, it is implemented as a 3D CNN
    with a kernel size of 1 over the time dimension.

    Parameters
    ----------
    input_channels : int
        Number of channels in the input images.
    n_blocks: int
        Number of downsampling blocks. Each block divides the spatial dimensions by 2.
    base_channels : int
        Number of channels in the first layer of the network. The number of channels is doubled
        in each downsampling block.
    base_block : str
        The type of block to use within the downsampling blocks. Must be either 'conv' or 'cbam'.
    kernel_size : int, optional
        The size of the kernel used in the 3D convolution. Defaults to 3.
    """

    def __init__(
        self, input_channels, n_blocks, base_channels, base_block, kernel_size=3
    ):
        super().__init__()

        self.input_channels = input_channels
        self.n_blocks = n_blocks
        self.base_channels = base_channels

        # Create the layers
        self.layers = nn.Sequential()
        in_channels = input_channels
        for i in range(n_blocks):
            out_channels = base_channels * 2**i
            self.layers.append(
                DownsamplingBlock3D(
                    in_channels, out_channels, base_block, (1, kernel_size, kernel_size)
                )
            )
            in_channels = out_channels

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, T, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, C, T, H', W').
        """
        return self.layers(x)

    def output_size(self, input_size):
        """
        Computes the output size of the network given an input size.

        Parameters
        ----------
        input_size : tuple
            Size of the input tensor, as a tuple (C, H, W).
        
        Returns
        -------
        tuple
            Size of the output tensor, as a tuple (C', H', W').
        """
        # Compute the size after each downsampling block
        size = (input_size[0], 1, input_size[1], input_size[2])
        for layer in self.layers:
            size = layer.output_size(size)

        return (size[0], size[2], size[3])
