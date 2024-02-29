from torch import nn
from models.cnn3d import DownsamplingBlock3D


class Encoder3d(nn.Module):
    """
    3D Convolutional encoder. Input is a 5D tensor (N, C, D, H, W).
    The encoder is a sequence of 3D Downsampling blocks. Each block divides
    the height and width by 2 and doubles the number of channels.

    The encoder has D branches, which treat the temporal / depth dimension
    with different kernel sizes (2 to D). Each branch performs 3D convolutions
    without padding the depth dimension, until the depth is 1.
    A branch with depth kernel size k thus has

    Parameters
    ----------
    input_shape: tuple of int (C, D, H, W)
        The shape of the input, channels first.
    base_block: str
        'conv' or 'cbam', type of block to use.
    conv_blocks: int, optional
        Number of convolutional blocks.
    hidden_channels: int, optional
        Number of hidden channels in the first convolutional layer.
    """

    def __init__(self, input_shape, base_block, conv_blocks=5, hidden_channels=4):
        super().__init__()
        self.base_block = base_block
        input_channels, d, h, w = input_shape
        # Define the list of the number of channels in the output of each block
        # Begin with the input channels
        self.output_channels = [input_channels]
        c = input_channels
        # Create the successive convolutional blocks
        self.conv_blocks = nn.ModuleList([])
        for i in range(0, conv_blocks):
            new_c = 2**i * hidden_channels
            # Save the number of channels in the output of the block
            self.output_channels.append(new_c)
            # Create the block
            # The kernel size is 3x3x3 for all blocks. The kernel size for the
            # depth dimension is:
            # - 2 if the depth is > 1
            # - 1 if the depth is 1
            kernel_size = (2, 3, 3) if d > 1 else (1, 3, 3)
            self.conv_blocks.append(
                DownsamplingBlock3D(c, new_c, base_block, kernel_size)
            )  # DxHxW -> D/2xH/2xW/2
            c = new_c
            # Keep track of the output size of each block
            d, h, w = self.conv_blocks[-1].output_size((d, h, w))
        # Add a global average pooling layer
        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.output_shape = (c, 1, 1, 1)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch tensor of dimensions (N, C, D, H, W)
            Input batch.
        Returns
        -------
        x: torch tensor of dimensions (N, C, 1, 1, 1)
            Output batch.
        """
        # Apply the convolutional blocks
        for block in self.conv_blocks:
            x = block(x)
        # Apply the global pooling
        x = self.global_pooling(x)
        return x
