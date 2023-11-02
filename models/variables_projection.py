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
    input_shape : int
        Number of input channels.
    output_shape : int
        Size of the desired 3D output under the form (C, D, H, W).
    """
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        # Input batch normalization
        self.input_batchnorm = nn.BatchNorm1d(input_shape)
        # Linear layer to project to a vector of length 1 * D * H * W
        self.linear = nn.Linear(input_shape, output_shape[1] * output_shape[2] * output_shape[3])
        # Reshape the output to (1, D, H, W) and apply batch normalization
        self.batchnorm = nn.BatchNorm3d(1)
        # 3D Conv layers to project to the desired output shape (C, D, H, W)
        n_output_channels = output_shape[0]
        self.conv1 = nn.Conv3d(1, n_output_channels, kernel_size=3, padding=1)
        self.conv_head = nn.Conv3d(n_output_channels, n_output_channels, kernel_size=1)

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
        # Apply batch normalization to the input
        x = self.input_batchnorm(x)
        # Project to a vector of length 1 * D * H * W
        x = torch.selu(self.linear(x))
        # Reshape to (1, D, H, W) and apply batch normalization
        x = x.view(x.shape[0], 1, self.output_shape[1], self.output_shape[2], self.output_shape[3])
        x = self.batchnorm(x)
        # Apply 3D convolutions to project to a tensor of shape (C, D, H, W)
        x = torch.selu(self.conv1(x))
        x = self.conv_head(x)

        return x 
