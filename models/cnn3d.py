"""
Implements a single 3D CNN for various tasks.
"""
import torch
import torch.nn as nn
from math import prod


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


class ConvBlock3D(nn.Module):
    """
    A single convolutional block.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple of ints, optional
        Size of the convolutional kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=1, stride=2)
        self.batchnorm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = torch.selu(self.conv1(x))
        x = torch.selu(self.conv2(x))
        x = self.batchnorm(x)
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
        d = conv_layer_output_size(d, self.kernel_size, padding=1)
        h = conv_layer_output_size(h, self.kernel_size, padding=1)
        w = conv_layer_output_size(w, self.kernel_size, padding=1)
        d = conv_layer_output_size(d, self.kernel_size, padding=1, stride=2)
        h = conv_layer_output_size(h, self.kernel_size, padding=1, stride=2)
        w = conv_layer_output_size(w, self.kernel_size, padding=1, stride=2)
        return (d, h, w)


class CNN3D(nn.Module):
    """
    A simple 3D CNN for various tasks.
    
    Parameters
    ----------
    input_size: tuple (int, int, int)
        Size of the input cubes under the form (D, H, W).
    input_channels : int
        Number of input channels (e.g. reanalysis, satellite, etc.).
    output_shape: int or tuple of ints
        Shape of the output, not accounting for the batch size.
        If an integer, size of the output vector.
    conv_blocks: int, optional
        Number of convolutional blocks.
    hidden_channels: int, optional
        Number of hidden channels in the first convolutional layer.
    """
    def __init__(self, input_size, input_channels, output_shape,
                 conv_blocks=4, hidden_channels=4):
        super().__init__()
        d, h, w = input_size
        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        self.output_shape = output_shape
        # Add a batch normalization layer at the beginning
        self.batchnorm = nn.BatchNorm3d(input_channels)
        # Input convolutional block
        c = max(hidden_channels, input_channels)
        self.input_conv = nn.Conv3d(input_channels, c, kernel_size=3, padding=1) # DxHxW -> DxHxW
        # Create the successive convolutional blocks
        self.conv_blocks = nn.ModuleList([])
        d, h, w = input_size
        for i in range(conv_blocks):
            new_c = (i + 1) * hidden_channels
            self.conv_blocks.append(ConvBlock3D(c, new_c))
            c = new_c
            # Keep track of the output size of each block
            d, h, w = self.conv_blocks[-1].output_size((d, h, w))
        # Linear prediction head
        self.fc1 = nn.Linear(c * d * h * w, 128)
        self.fc2 = nn.Linear(128, 64)
        output_size = prod(output_shape)
        self.fc3 = nn.Linear(64, output_size)


    def forward(self, x):
        """
        Parameters
        ----------
        x: torch tensor of dimensions (N, C, D, H, W)
            Input batch.
        Returns
        -------
        torch tensor of dimensions (N, output_size)
            Output batch of N intensity predictions.
        """
        # Apply batch normalization to the input
        x = self.batchnorm(x)
        # Apply the input convolutional block
        x = torch.selu(self.input_conv(x))
        # Apply the successive convolutional blocks
        for block in self.conv_blocks:
            x = block(x)
        # Flatten the output
        x = x.view(x.shape[0], -1)
        # Apply the prediction head
        x = torch.selu(self.fc1(x))
        x = torch.selu(self.fc2(x))
        x = self.fc3(x)
        # Reshape x to the desired output shape
        return x.view(x.shape[0], *self.output_shape)

