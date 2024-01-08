from torch import nn
from models.cnn3d import DownsamplingBlock3D


class Encoder3d(nn.Module):
    """
    A simple 3D CNN Encoder. 
    
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
    def __init__(self, input_shape, base_block,
                 conv_blocks=7, hidden_channels=4):
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
            new_c = 2 ** i * hidden_channels
            # Save the number of channels in the output of the block
            self.output_channels.append(new_c)
            self.conv_blocks.append(DownsamplingBlock3D(c, new_c, base_block)) # DxHxW -> DxH/2xW/2
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
        A list of torch tensors of dimensions (N, C, D, H, W)
            The output of each block.
        """
        # Apply the convolutional blocks
        outputs = []
        for block in self.conv_blocks:
            x = block(x)
            outputs.append(x)
        return outputs
