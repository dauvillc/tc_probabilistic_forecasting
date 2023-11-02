"""
Implements a single 3D CNN for various tasks.
"""
import torch
import torch.nn as nn
from models.variables_projection import VectorProjection3D


class CNN3D(nn.Module):
    """
    A simple 3D CNN for various tasks.
    
    Parameters
    ----------
    input_size: tuple (int, int, int)
        Size of the input cubes under the form (D, H, W).
    input_channels : int
        Number of input channels (e.g. reanalysis, satellite, etc.).
    input_variables: int
        Number of input variables I (vector containing scalar variables, such
        as latitude and longitude).
        Can be zero.
    output_size: int
        Number of output channels.
    hidden_channels: int, optional
        Number of hidden channels in the first convolutional layer.
    """
    def __init__(self, input_size, input_channels, input_variables, output_size, hidden_channels=4):
        super().__init__()
        d, h, w = input_size
        # If there are input variables, add a vector projection layer
        self.input_variables = input_variables
        if input_variables > 0:
            self.vector_projection = VectorProjection3D(input_variables, (input_channels, d, h, w))
        # Add a batch normalization layer at the beginning
        self.batchnorm = nn.BatchNorm3d(input_channels)
        # Convolutional blocks:
        # Each block is composed of 2 convolutional layers with a kernel size
        # of 3 and a padding of 1, followed by a batch normalization layer.
        c = hidden_channels
        self.conv1_1 = nn.Conv3d(input_channels, c, kernel_size=3, padding=1) # Tx11x11 -> Tx11x11
        # The second layer of the block has a stride of 2, instead of using
        # max pooling
        self.conv1_2 = nn.Conv3d(c, c * 2, kernel_size=3, stride=2) # Tx11x11 -> Tx5x5
        self.conv1_batchnorm = nn.BatchNorm3d(c * 2)
        self.conv2_1 = nn.Conv3d(c * 2, c * 2, kernel_size=3, padding=1) # Tx5x5 -> 5x5
        self.conv2_2 = nn.Conv3d(c * 2, c * 4, kernel_size=3, padding=1, stride=2) # Tx5x5 -> 3x3
        self.conv2_batchnorm = nn.BatchNorm3d(c * 4)
        # Fully connected layers
        self.fc1 = nn.Linear(c * 4 * 3 * 3, c * 4)
        self.fc2 = nn.Linear(c * 4, output_size)

    def forward(self, past_images, past_variables=None):
        """
        Parameters
        ----------
        past_images : torch tensor of dimensions (N, C, D, H, W)
            Input batch. 
        past_variables: torch vector of shape (N, I), optional.
            Input batch of scalar variables.
        Returns
        -------
        torch tensor of dimensions (N, output_size)
            Output batch of N intensity predictions.
        """
        # If there are input variables, project them to a tensor of shape
        # (C, D, H, W)
        if past_variables is not None:
            if self.input_variables != past_variables.shape[1]:
                raise ValueError(f"Expected {self.input_variables} input variables, "
                                 f"got {past_variables.shape[1]}")
            projection = self.vector_projection(past_variables)
            # Sum the input images and the input variables if there are any
            x = past_images + projection
        else:
            x = past_images
        # Apply batch normalization
        x = self.batchnorm(x)
        # Apply the successive convolutional blocks, with selu as the activation
        # function
        x = torch.selu(self.conv1_1(x))
        x = torch.selu(self.conv1_2(x))
        x = self.conv1_batchnorm(x)
        x = torch.selu(self.conv2_1(x))
        x = torch.selu(self.conv2_2(x))
        x = self.conv2_batchnorm(x)
        # Flatten the output
        x = x.view(x.shape[0], -1)
        # Apply the fully connected layers
        x = torch.selu(self.fc1(x))
        x = self.fc2(x)
        return x

