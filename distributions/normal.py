"""
Implements the NormalDistribution class, which can be used as output distribution P(y|x).
"""
import torch
from torch.distributions import Normal


def normal_crps(mu, sigma, y):
    """
    Computes the CRPS for a normal distribution.

    Parameters
    ----------
    mu : torch.Tensor
        The predicted mean for each sample.
    sigma : torch.Tensor
        The predicted standard deviation for each sample.
    y : torch.Tensor
        The true values for each sample.

    Returns
    -------
    torch.Tensor of shape (N, 1)
        The CRPS for each sample.
    """
    normal_dist = Normal(torch.zeros_like(mu), torch.ones_like(sigma))
    std_y = (y - mu) / sigma
    crps = sigma * (std_y * (2 * normal_dist.cdf(std_y) - 1)\
            + 2 * torch.exp(normal_dist.log_prob(std_y))\
            - 1 / torch.sqrt(torch.tensor(torch.pi)))
    return crps


def normal_mae(predicted_params, y):
    """
    Returns the MAE between the predicted mean (which is also the median) and the true value.
    """
    # Flatten the tensors to get rid of the time dimension
    predicted_params = predicted_params.view(-1, 2)
    y = y.view(-1)
    mu = predicted_params[:, 0]
    return torch.abs(mu - y).mean()


class NormalDistribution:
    """
    Object that contains the loss function, cdf and metrics for normal distributions.

    Parameters
    ----------
    tasks: dict
        Pointer to the tasks dictionary, which contains the normalization constants.
    """
    def __init__(self, tasks):
        self.tasks = tasks
        self.n_parameters = 2

        # Metrics
        # The MAE is the mean absolute error between the predicted mean (which is also the
        # median) and the true value.
        self.metrics = {
                'MAE': normal_mae,
                'CRPS': self.loss_function,
        }

    def inverse_cdf(self, predicted_params, u):
        """
        Computes the inverse of the CDF for a normal distribution.

        Parameters
        ----------
        predicted_params : torch.Tensor of shape (N, 2)
            The predicted parameters of the normal distribution for each sample.
        u : float 
            The probability at which the inverse CDF is computed.

        Returns
        -------
        torch.Tensor of shape (N, 1)
            The inverse CDF for each sample.
        """
        mu, sigma = predicted_params[:, 0], predicted_params[:, 1]
        normal_dist = Normal(mu, sigma)
        return normal_dist.icdf(torch.tensor(u))

    def loss_function(self, predicted_params, y):
        """
        Computes the CRPS for a normal distribution, whose parameters are predicted
        by the model.
        If the predicted sigma is negative, it is clamped to a small positive value, and
        the distance between 0 and sigma is added to the loss.

        Parameters
        ----------
        predicted_params : torch.Tensor of shape (N, T, 2)
            The predicted parameters of the normal distribution for each sample and time step.
        y: torch.Tensor of shape (N, T)
            The true values for each sample and time step.

        Returns
        -------
        float
            The mean CRPS over the batch.
        """
        # We first need to flatten the tensors to get rid of the time dimension
        predicted_params = predicted_params.view(-1, self.n_parameters)
        y = y.flatten()
        mu, sigma = predicted_params[:, 0], predicted_params[:, 1]
        # Where sigma is negative, we add the distance between 0 and sigma to the loss
        loss = torch.zeros_like(sigma)
        loss[sigma < 0] = -sigma[sigma < 0]
        # Clamp sigma to a small positive value
        sigma = torch.clamp(sigma, min=1e-5)
        # Compute the CRPS
        loss += normal_crps(mu, sigma, y)
        return loss.mean()

    def denormalize(self, predicted_params, task):
        """
        De-normalizes the predicted parameters of the distribution.

        Parameters
        ----------
        predicted_params : torch.Tensor of shape (N, T, 2)
            The predicted parameters of the normal distribution for each sample and time step.
        task : str
            The name of the task.

        Returns
        -------
        torch.Tensor of shape (N, T, 2)
            The de-normalized parameters.
        """
        # Get the normalization constants from the tasks dictionary
        means = torch.tensor(self.tasks[task]['means'].values, dtype=torch.float32, device=predicted_params.device)
        stds = torch.tensor(self.tasks[task]['stds'].values, dtype=torch.float32, device=predicted_params.device)
        # The normalization constants are of shape (T,) but the predicted means and stds
        # are of shape (N, T), so we need to add a new dimension to the constants.
        means = means.unsqueeze(0)
        stds = stds.unsqueeze(0)
        # De-normalize the predictions by rescaling the predicted mean and std
        # Mean of the distribution:
        predicted_params[:, :, 0] = predicted_params[:, :, 0] * stds + means
        # Standard deviation of the distribution:
        predicted_params[:, :, 1] = predicted_params[:, :, 1] * stds
        return predicted_params
        
    def hyperparameters(self):
        return {}
