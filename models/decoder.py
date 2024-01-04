import torch.nn as nn
from models.cnn3d import UpsampleConvBlock3D


class Decoder3d(nn.Module):
    """
    A simple decoder for 3D data.

    Parameters
    ----------
    in_channels: int
        Number of input channels.
    out_channels: int
        Number of output channels.
    base_block: str
        'conv' or 'cbam', type of block to use.
    n_blocks: int, optional
        Number of upsampling blocks.
    """
    def __init__(self, in_channels, out_channels, base_block, n_blocks):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.base_block = base_block

        # Build the blocks, with decreasing number of channels.
        # The number of channels is symmetric with the encoder.
        self.blocks = nn.ModuleList()
        channels = in_channels
        for i in range(n_blocks, 0, -1):
            channels = i * in_channels
            self.blocks.append(UpsampleConvBlock3D(in_channels, channels, base_block))
            in_channels = channels
        
        # Final convolution.
        self.final_conv = nn.Conv3d(channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x

    def output_shape(self, input_shape):
        """
        Computes the output shape of the decoder.

        Parameters
        ----------
        input_shape : tuple (int, int, int)
            Shape of the input under the form (D, H, W).
        """
        shape = input_shape
        for block in self.blocks:
            shape = block.output_size(shape)
        return shape

