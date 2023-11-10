"""
Implements modules to project scalar variables into a datacube.
"""
import torch
import torch.nn as nn


class VectorProjection3D(nn.Module):
    """
    Implements a vector projection layer.

    Parameters
    ----------
    input_len : int
        Length of the input vector.
    output_shape : int
        Size of the desired 3D output under the form (C, D, H, W).
    """
    def __init__(self, input_len, output_shape):
        super().__init__()
        self.input_len = input_len
        self.output_shape = output_shape
        c, d, h, w = output_shape
        # Input batch normalization
        self.input_batchnorm = nn.BatchNorm1d(input_len)
        # Linear layer to project to a vector of length 1 * H * W
        self.linear = nn.Linear(input_len, h * w)
        # Reshape the output to (1, H, W) and apply batch normalization
        self.batchnorm_1 = nn.BatchNorm2d(1)
        # 2D Conv layers to project to the desired output shape (D, H, W)
        self.conv1_1 = nn.Conv2d(1, d, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(d, d, kernel_size=3, padding=1)
        # Reshape the output to (1, D, H, W) and apply batch normalization
        self.batchnorm_2 = nn.BatchNorm3d(1)
        # 3D Conv layers to project to the desired output shape (C, D, H, W)
        n_output_channels = output_shape[0]
        self.conv2_1 = nn.Conv3d(1, n_output_channels, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv3d(n_output_channels, n_output_channels, kernel_size=1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch tensor of dimensions (N, C, H, W)
            Input batch of N patches of dimensions (C, H, W).

        Returns
        -------
        torch tensor of dimensions (N, C, D, H, W)
            Output batch of N patches of dimensions (C, D, H, W).
        """
        c, d, h, w = self.output_shape
        # Apply batch normalization to the input
        x = self.input_batchnorm(x)
        # Project to a vector of length H * W
        x = torch.selu(self.linear(x))
        # Reshape to (N, 1, H, W) and apply batch normalization
        x = x.view(-1, 1, h, w)
        x = self.batchnorm_1(x)
        # 2D Conv layers
        x = torch.selu(self.conv1_1(x))
        x = torch.selu(self.conv1_2(x))
        # Reshape to (N, 1, D, H, W) and apply batch normalization
        x = x.view(-1, 1, d, h, w)
        x = self.batchnorm_2(x)
        # 3D Conv layers
        x = torch.selu(self.conv2_1(x))
        x = self.conv2_2(x)

        return x 
