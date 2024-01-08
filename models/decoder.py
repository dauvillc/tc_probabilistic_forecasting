import torch.nn as nn
import torch
from models.cnn3d import UpsampleConvBlock3D


class Decoder3d(nn.Module):
    """
    A simple decoder for 3D data.

    Parameters
    ----------
    encoder: Encoder3d
        The encoder used to produce the latent space. Used to infer the number
        of channels in the input of the decoder.
    out_channels: int
        Number of output channels.
    base_block: str
        'conv' or 'cbam', type of block to use.
    n_blocks: int, optional
        Number of upsampling blocks.
    """
    def __init__(self, encoder, out_channels, base_block, n_blocks):
        super().__init__()
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.base_block = base_block
        # The number of input channels is the number of channels in the output
        # of the last block of the encoder
        self.in_channels = encoder.output_channels[-1]
        
        # Create the first upsampling block that takes the latent space as input
        # The ouput channels are the same as the input channels of the last block
        # of the encoder
        self.blocks = nn.ModuleList([])
        block_out_channels = encoder.output_channels[-2]
        block_in_channels = self.in_channels
        self.blocks.append(UpsampleConvBlock3D(block_in_channels, block_out_channels, base_block))
        # Create the intermediate upsampling blocks
        for i in range(1, n_blocks - 1):
            # The input channels are the output channels of the previous block
            # plus the number of channels in the output of the corresponding
            # block of the encoder (skip connection)
            block_in_channels = 2 * block_out_channels
            # The output channels are the same as the input channels of the
            # corresponding block of the encoder
            block_out_channels = encoder.output_channels[-i - 2]
            self.blocks.append(UpsampleConvBlock3D(block_in_channels, block_out_channels, base_block))
        # Create the last upsampling block
        self.blocks.append(UpsampleConvBlock3D(block_out_channels * 2,
                                               out_channels,
                                               base_block))
        # Final convolutions
        self.final_conv = nn.Conv3d(out_channels, out_channels, kernel_size=1)
    
    def forward(self, latent_space, encoder_outputs):
        """
        Parameters
        ----------
        latent_space: torch tensor of dimensions (N, C, T, H, W)
            The latent space to use as input of the decoder.
        encoder_outputs: list of torch tensors of dimensions (N, c_i, T, h_i, w_i)
            The outputs of the encoder (skip connections).
        """
        # Apply the first upsampling block
        x = self.blocks[0](latent_space)
        # Apply the rest of the upsampling blocks to the concatenation of the
        # output of the previous block and the corresponding output of the encoder
        for i in range(1, self.n_blocks):
            x = torch.cat([x, encoder_outputs[-i - 1]], dim=1)
            x = self.blocks[i](x)
        # Apply the final convolution
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

