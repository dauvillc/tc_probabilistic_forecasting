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


class BasicCNNBlock3D(nn.Module):
    """
    A single block of a 3D CNN whose output size is the same as its input size.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the kernel. The padding is computed automatically to keep the
        spatial size constant.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # Compute padding to keep the spatial size constant
        padding = (kernel_size - 1) // 2
        # Convolution
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = torch.selu(self.conv(x))
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
        d, h, w = input_size
        h = conv_layer_output_size(h, self.conv.kernel_size[1], padding=self.conv.padding[1])
        w = conv_layer_output_size(w, self.conv.kernel_size[2], padding=self.conv.padding[2])
        return (d, h, w)


class DownsamplingBlock3D(nn.Module):
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
    """
    def __init__(self, in_channels, out_channels, base_block):
        super().__init__()
        # Entry convolution
        self.entry_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        # Base block
        if base_block == 'cbam':
            self.base_block = CBAM3D(out_channels)
        elif base_block == 'conv':
            self.base_block = BasicCNNBlock3D(out_channels, out_channels, kernel_size=3)
        else:
            raise ValueError(f'Unknown base block: {base_block}')
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
    base_block: str
        'conv' or 'cbam', type of base block to use before downsampling.
    """
    def __init__(self, in_channels, out_channels, base_block):
        super().__init__()
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        # Entry convolution
        self.entry_conv = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        # Base block
        if base_block == 'cbam':
            self.base_block = CBAM3D(in_channels)
        elif base_block == 'conv':
            self.base_block = BasicCNNBlock3D(in_channels, in_channels, kernel_size=3)
        else:
            raise ValueError(f'Unknown base block: {base_block}')
        # Output convolutions: 1x3x3 followed by 1x1x1
        self.output_conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.output_conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = torch.selu(self.entry_conv(x))
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
