"""
Implements the TemporalBlock and TemporalEncoder classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """
    Encodes a sequence of frames into a latent representation, using a combination
    of local convolutions and global poolings.
    Based on Hu et. al. https://arxiv.org/pdf/2003.06409.pdf (2020).

    Parameters
    ----------
    in_channels: int
        Number of input channels.
    height: int
        Height of the input frames.
    width: int
        Width of the input frames.
    kernel_size_time: int
        Kernel size over the time dimension.
    kernel_size_space: int
        Kernel size over the space dimensions.
    """

    def __init__(self, in_channels, height, width, kernel_size_time, kernel_size_space):
        super().__init__()
        # Local context
        # First, a 1x1x1 conv to reduce the number of channels
        self.local_compressors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(in_channels, in_channels // 2, kernel_size=1),
                    nn.SELU(),
                    nn.BatchNorm3d(in_channels // 2),
                )
                for _ in range(1)
            ]
        )
        # Spatio-temporal conv
        convs = nn.ModuleList([])
        kt, ks = kernel_size_time, kernel_size_space
        c = in_channels // 2
        convs.append(
            nn.Conv3d(c, c, kernel_size=(kt, ks, ks), padding="same")
        )
        # Assemble the local convolutions with SELU and batch normalization
        self.local_convs = nn.ModuleList(
            [nn.Sequential(conv, nn.SELU(), nn.BatchNorm3d(c)) for conv in convs]
        )
        # Global context
        self.pools = nn.ModuleList([])
        self.pools.append(nn.AvgPool3d(kernel_size=(kt, height, width)))
        self.pools.append(nn.AvgPool3d(kernel_size=(1, height // 2, width // 2)))
        self.pools.append(nn.AvgPool3d(kernel_size=(1, height // 4, width // 4)))
        # Channel reduction for the global context
        self.global_compressors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(in_channels, in_channels // 3, kernel_size=1),
                    nn.SELU(),
                    nn.BatchNorm3d(in_channels // 3),
                )
                for _ in range(3)
            ]
        )
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv3d(1 * (in_channels // 2) + 3 * (in_channels // 3), in_channels, kernel_size=1),
            nn.SELU(),
            nn.BatchNorm3d(in_channels),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (N, C, T, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, C', T, H, W).
        """
        # List of all tensors obtained across the local and global contexts
        middle_tensors = []
        # Local context
        # Apply the local compressors
        compressed = [compressor(x) for compressor in self.local_compressors]
        # Apply each type of motion convolution
        middle_tensors += [conv(z) for conv, z in zip(self.local_convs, compressed)]
        # Global context
        # Apply the global poolings
        pooled = [pool(x) for pool in self.pools]
        # Apply the global compressors
        compressed = [
            compressor(pool) for compressor, pool in zip(self.global_compressors, pooled)
        ]
        # Upsample the global context tensors
        middle_tensors += [
            F.interpolate(z, size=(x.size(2), x.size(3), x.size(4))) for z in compressed
        ]
        # Concatenate all tensors
        middle_tensors = torch.cat(middle_tensors, dim=1)
        # Final convolution
        out = self.final_conv(middle_tensors)
        # Skip connection with the input tensor
        out += x
        return out


class TemporalEncoder(nn.Module):
    """
    Assembles multiple TemporalBlock modules to encode a sequence of frames.

    Parameters
    ----------
    in_channels: int
        Number of input channels.
    height: int
        Height of the input frames.
    width: int
        Width of the input frames.
    out_channels: int
        Number of output channels.
    kernel_size_time: int
        Kernel size over the time dimension.
    kernel_size_space: int
        Kernel size over the space dimensions.
    num_blocks: int
        Number of TemporalBlock modules to use.
    """

    def __init__(
        self,
        in_channels,
        height,
        width,
        out_channels,
        kernel_size_time,
        kernel_size_space,
        num_blocks,
    ):
        super().__init__()
        # The blocks don't modify the number of channels
        self.blocks = nn.ModuleList(
            [
                TemporalBlock(
                    in_channels,
                    height,
                    width,
                    kernel_size_time,
                    kernel_size_space,
                )
                for _ in range(num_blocks)
            ]
        )
        # Add a 1x1x1 conv to compress the number of channels
        self.compress_channels = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (N, C, T, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, C', T, H, W).
        """
        for block in self.blocks:
            x = block(x)
        x = torch.selu(self.compress_channels(x))
        return x
