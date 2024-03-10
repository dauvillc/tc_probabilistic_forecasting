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
    The latent space is flattened and fed to a linear layer. The output is then
    concatenated with the embedded past variables, and fed to a second linear
    layer.

    Parameters
    ----------
    input_shape: tuple of int (c, d, h, w)
        The shape of the latent space produced by the encoder.
    output_depth: int
        Size T of the depth dimension where the output size is
        c * T * h * w.
    n_input_vars: int
        The number of past variables.
    hidden_size_reduction: int
        The reduction factor in the hidden layers.
    """
    def __init__(self, input_shape, output_depth,
                 n_input_vars, hidden_size_reduction):
        super().__init__()
        # Compute the size of the latent space
        c, d, h, w = input_shape
        self.input_size = c * d * h * w
        # Compute the size of the hidden layers
        self.hidden_size = self.input_size // hidden_size_reduction
        # Compute the size of the output
        self.output_size = c * output_depth * h * w
        # Linear layers for embedding the past variables
        self.embedding_1 = nn.Linear(n_input_vars, n_input_vars)
        self.embedding_2 = nn.Linear(n_input_vars, n_input_vars * 2)
        # Build the linear layers
        self.linear_1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear_2 = nn.Linear(self.hidden_size + n_input_vars * 2, self.hidden_size)
        self.linear_3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, latent_space, past_vars):
        """
        Parameters
        ----------
        latent_space: torch tensor of dimensions (N, input_channels, ...)
            The latent space produced by the encoder.
        past_vars: Mapping of str to torch tensor
            The past variables. The keys are the variable names, and the values
            are torch tensors of dimensions (N, P) where P is the number of past
            steps.
        Returns
        -------
        torch tensor of dimensions (N, output_channels)
            The output of the linear module.
        """
        # Concatenate the past variables into a single tensor
        past_vars = torch.cat(list(past_vars.values()), dim=1)
        # Flatten the latent space
        latent_space = latent_space.reshape(latent_space.shape[0], -1) # (N, input_size)
        # Embed the past variables
        past_vars = torch.selu(self.embedding_1(past_vars))
        past_vars = torch.selu(self.embedding_2(past_vars))
        # Apply the linear layers
        x = torch.selu(self.linear_1(latent_space))
        x = torch.cat([x, past_vars], dim=1)
        x = torch.selu(self.linear_2(x))
        x = torch.selu(self.linear_3(x))
        return x


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
    n_output_vars: int
        The number of predicted variables.
    future_steps: int
        The number of future steps to predict.
    """
    def __init__(self, input_size, n_output_vars, future_steps):
        super().__init__()
        self.n_output_vars = n_output_vars
        self.future_steps = future_steps
        self.output_size = n_output_vars * future_steps
        # Fully connected layer
        self.fc = nn.Linear(input_size, self.output_size)

    def forward(self, latent_space):
        """
        Parameters
        ----------
        latent_space: torch tensor of dimensions (N, input_size)
            The latent space produced by the common linear module.
        Returns
        -------
        torch tensor of dimensions (N, future_steps, n_output_vars)
            The prediction.
        """
        # Apply the fully connected layer
        prediction = self.fc(latent_space)
        # Reshape the prediction
        return prediction.reshape(-1, self.future_steps, self.n_output_vars)


class MultivariatePredictionHead(nn.Module):
    """
    Similar to PredictionHead, but predicts the parameters of a multivariate
    distribution over all time steps at once, instead of predicting the parameters
    of the marginal distribution at each time step.

    Parameters
    ----------
    input_size: int
        The size of the input latent space.
    n_parameters:
        The number of parameters of the multivariate distribution.
    """
    def __init__(self, input_size, n_parameters):
        super().__init__()
        self.n_parameters = n_parameters
        self.output_size = n_parameters
        # Fully connected layer
        self.fc = nn.Linear(input_size, self.n_parameters)

    def forward(self, latent_space):
        """
        Parameters
        ----------
        latent_space: torch tensor of dimensions (N, input_size)
            The latent space produced by the common linear module.
        Returns
        -------
        torch tensor of dimensions (N, n_parameters)
            The prediction.
        """
        # Apply the fully connected layer
        prediction = self.fc(latent_space)
        return prediction
