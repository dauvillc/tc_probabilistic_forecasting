"""
Implements a single 3D CNN for various tasks.
"""

import torch
import torch.nn as nn
from models.cbam import CBAM3D


def conv_layer_output_size(input_size, kernel_size, padding=0, stride=1):
    """
    Computes the output size of a convolutional layer.

    Parameters
    ----------
    input_size : int
        Size of the input.
    kernel_size : int
        Size of the kernel.
    padding : int, optional
        Padding size.
    stride : int, optional
        Stride size.
    """
    return (input_size + 2 * padding - kernel_size) // stride + 1


class BasicCNNBlock2D(nn.Module):
    """
    A single block of a 2D CNN whose output size is the same as its input size.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the kernel over the H and W dimensions.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        # Order based on "Identity Mappings in Deep Residual Networks"
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SELU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm2d(out_channels),
            nn.SELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
        )

    def forward(self, x):
        return self.layers(x)

    def output_size(self, input_size):
        """
        Computes the output size of the block.

        Parameters
        ----------
        input_size : tuple (int, int, int)
            Size of the input under the form (C, H, W).
        """
        return (self.out_channels, *input_size[1:])


class DownsamplingBlock2D(nn.Module):
    """
    A single block that downsamples the height and width of the input tensor.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    base_block: str
        'conv' or 'cbam'. Type of base block to use before downsampling.
    kernel_size : int
        Spatial size of the kernel.
    """

    def __init__(self, in_channels, out_channels, base_block, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        # Entry convolution.
        self.entry_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        # Base block
        if base_block == "cbam":
            self.base_block = CBAM3D(out_channels)
        elif base_block == "conv":
            self.base_block = BasicCNNBlock2D(out_channels, out_channels, kernel_size=kernel_size)
        else:
            raise ValueError(f"Unknown base block: {base_block}")
        # Apply a convolution with a stride of 2 to downsample
        self.final_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=(2, 2),
            padding=(1, 1),
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.selu(self.entry_conv(x))
        x = x + self.base_block(x)
        x = torch.selu(self.final_conv(x))
        x = self.batch_norm(x)
        return x

    def output_size(self, input_size):
        """
        Computes the output size of the block.

        Parameters
        ----------
        input_size : tuple (int, int, int)
            Size of the input under the form (C, H, W).
        """
        _, h, w = self.base_block.output_size(input_size)
        h = conv_layer_output_size(h, self.kernel_size, padding=1, stride=2)
        w = conv_layer_output_size(w, self.kernel_size, padding=1, stride=2)
        return (self.out_channels, h, w)



