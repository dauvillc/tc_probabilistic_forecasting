"""
Implements the linear modules:
- CommonLinearModule
- PredictionHead
"""
import torch
from torch import nn


class CommonLinearModule(nn.Module):
    """
    Implements a linear layer that receives as input the latent space produced by
    the encoder, and the past variables.
    The variables are projected into a latent space, and then concatenated with
    the latent space produced by the encoder.

    Parameters
    ----------
    input_shape: tuple of int (c, d, h, w)
        The shape of the latent space produced by the encoder.
    n_input_vars: int
        The number of past variables.
    output_size: int
        The size of the output latent space.
    """
    def __init__(self, input_shape, n_input_vars, output_size):
        super().__init__()
        c, d, h, w = input_shape
        # Linear layer for projecting the variables
        # Since the projection will be concatenated with the latent space,
        # its size doesn't need to be large.
        self.var_projection = nn.Linear(n_input_vars, n_input_vars)
        # Linear layer for projecting the latent space, concatenated with the
        # projected variables
        self.latent_projection = nn.Linear(c * d * h * w + n_input_vars, output_size)

    def forward(self, latent_space, variables):
        """
        Parameters
        ----------
        latent_space: torch tensor of dimensions (N, c, d, h, w)
            The latent space produced by the encoder.
        variables: torch tensor of dimensions (N, n_input_vars)
            The past variables.
        Returns
        -------
        torch tensor of dimensions (N, output_size)
            The latent space for the prediction heads.
        """
        # Project the variables
        projected_vars = torch.selu(self.var_projection(variables))
        # Flatten the latent space
        flattened_latent_space = latent_space.flatten(start_dim=1)
        # Concatenate the latent space and the projected variables
        concatenated = torch.cat([flattened_latent_space, projected_vars], dim=1)
        # Project the concatenated tensor
        return torch.selu(self.latent_projection(concatenated))


class PredictionHead(nn.Module):
    """
    Implements a prediction head for a given task.
    The prediction head receives as input the latent space produced by the
    common linear module, and outputs a prediction for the given task at
    each future step.

    Parameters
    ----------
    input_size: int
        The size of the input latent space.
    output_size: int
        The number of predicted variables.
    future_steps: int
        The number of future steps to predict.
    """
    def __init__(self, input_size, output_size, future_steps):
        super().__init__()
        self.output_size = output_size
        self.future_steps = future_steps
        self.linear = nn.Linear(input_size, output_size * future_steps)

    def forward(self, latent_space):
        """
        Parameters
        ----------
        latent_space: torch tensor of dimensions (N, input_size)
            The latent space produced by the common linear module.
        Returns
        -------
        torch tensor of dimensions (N, future_steps, output_size)
            The prediction.
        """
        # Apply the linear layer
        prediction = self.linear(latent_space)
        # Reshape the prediction
        return prediction.reshape(-1, self.future_steps, self.output_size)
