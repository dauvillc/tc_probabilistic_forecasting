"""
Clément Dauvilliers - October 25th 2023
Implements a single 3D CNN for various tasks.
"""
import torch
import torch.nn as nn


class CNN3D(nn.Module):
    """
    A simple 3D CNN for various tasks.
    
    Parameters
    ----------
    input_channels : int, optional
        Number of input channels. The default is 69 (5 atmospheric variables
        at 13 levels + 4 surface variables).
    output_size: int, optional
        Number of output channels. The default is 1.
    """
    def __init__(self, input_channels=69, output_size=1):
        super().__init__()
        # Add a batch normalization layer at the beginning
        self.batchnorm = nn.BatchNorm3d(input_channels)
        # Convolutional blocks:
        # Each block is composed of 2 convolutional layers with a kernel size
        # of 3 and a padding of 1, followed by a batch normalization layer.
        self.conv1_1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1) # Tx11x11 -> Tx11x11
        # The second layer of the block has a stride of 2, instead of using
        # max pooling
        self.conv1_2 = nn.Conv3d(32, 64, kernel_size=3, stride=2) # Tx11x11 -> Tx5x5
        self.conv1_batchnorm = nn.BatchNorm3d(64)
        self.conv2_1 = nn.Conv3d(64, 64, kernel_size=3, padding=1) # Tx5x5 -> 5x5
        self.conv2_2 = nn.Conv3d(64, 128, kernel_size=3, padding=1, stride=2) # Tx5x5 -> 3x3
        self.conv2_batchnorm = nn.BatchNorm3d(128)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch tensor of dimensions (N, C, H, W)
            Input batch of N patches of dimensions (C, H, W).

        Returns
        -------
        torch tensor of dimensions (N, output_size)
            Output batch of N intensity predictions.
        """
        # Input normalization
        x = self.batchnorm(x)
        # Convolutional blocks
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_batchnorm(x)
        x = nn.functional.selu(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_batchnorm(x)
        x = nn.functional.selu(x)
        # Fully connected layers
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = nn.functional.selu(x)
        x = self.fc2(x)

        return x


