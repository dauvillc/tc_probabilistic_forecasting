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


class DownsamplingBlock3D(nn.Module):
    """
    A single block that downsamples the height and width of the input tensor.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    base_block: str, optional
        Type of base block to use before downsampling. For now, only
        'cbam' is supported.
    """
    def __init__(self, in_channels, out_channels, base_block='cbam'):
        super().__init__()
        # Entry convolution
        self.entry_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        # Base block
        self.base_block = CBAM3D(out_channels)
        # Apply a convolution with a stride of 2 to downsample
        self.final_conv = nn.Conv3d(out_channels, out_channels,
                                    kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.batch_norm = nn.BatchNorm3d(out_channels)
    
    def forward(self, x):
        x = torch.selu(self.entry_conv(x))
        x = self.base_block(x)
        x = torch.selu(self.final_conv(x))
        x = self.batch_norm(x)
        return x
    
    def output_size(self, input_size):
        """
        Computes the output size of the block.
        
        Parameters
        ----------
        input_size : tuple (int, int, int)
            Size of the input under the form (D, H, W).
        """
        d, h, w = self.base_block.output_size(input_size)
        h = conv_layer_output_size(h, 3, padding=1, stride=2)
        w = conv_layer_output_size(w, 3, padding=1, stride=2)
        return (d, h, w)


class UpsampleConvBlock3D(nn.Module):
    """
    A single block that upsamples the height and width of the input tensor.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    base_block: str, optional
        Type of base block to use before downsampling. For now, only
        'cbam' is supported.
    """
    def __init__(self, in_channels, out_channels, base_block='cbam'):
        super().__init__()
        # Entry convolution
        self.entry_conv = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        # Base block
        self.base_block = CBAM3D(in_channels)
        # Output convolutions: 1x3x3 followed by 1x1x1
        self.output_conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.output_conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = torch.selu(self.entry_conv(x))
        x = self.upsample(x)
        x = self.base_block(x)
        x = torch.selu(self.output_conv1(x))
        x = torch.selu(self.output_conv2(x))
        x = self.batch_norm(x)
        return x

    def output_size(self, input_size):
        """
        Computes the output size of the block.

        Parameters
        ----------
        input_size : tuple (int, int, int)
            Size of the input under the form (D, H, W).
        """
        d, h, w = self.base_block.output_size(input_size)
        return (d, h * 2, w * 2)
