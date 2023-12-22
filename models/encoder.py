import torch
from torch import nn
from models.cnn3d import DownsamplingBlock3D


class Encoder3d(nn.Module):
    """
    A simple 3D CNN Encoder. 
    
    Parameters
    ----------
    input_shape: tuple of int (C, D, H, W)
        The shape of the input, channels first.
    conv_blocks: int, optional
        Number of convolutional blocks.
    hidden_channels: int, optional
        Number of hidden channels in the first convolutional layer.
    """
    def __init__(self, input_shape,
                 conv_blocks=7, hidden_channels=4):
        super().__init__()
        input_channels, d, h, w = input_shape
        # Input convolutional block
        c = max(hidden_channels, input_channels)
        self.input_conv = nn.Conv3d(input_channels, c,
                                    kernel_size=(1, 3, 3), padding=(0, 1, 1)) # DxHxW -> DxHxW
        # Create the successive convolutional blocks
        self.conv_blocks = nn.ModuleList([])
        for i in range(conv_blocks):
            new_c = (i + 1) * hidden_channels
            self.conv_blocks.append(DownsamplingBlock3D(c, new_c)) # DxHxW -> DxH/2xW/2
            c = new_c
            # Keep track of the output size of each block
            d, h, w = self.conv_blocks[-1].output_size((d, h, w))
        self.output_shape = (c, d, h, w)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch tensor of dimensions (N, C, D, H, W)
            Input batch.
        Returns
        -------
        torch tensor of dimensions (N, c, d, h, w)
            Output batch, in the latent space.
        """
        # Apply the input convolutional block
        x = torch.selu(self.input_conv(x))
        # Apply the successive convolutional blocks
        for block in self.conv_blocks:
            x = block(x)
        return x
