"""
Implements the NormalDistribution class, which can be used as output distribution P(y|x).
"""
import torch
from torch.distributions import Normal
from utils.utils import add_batch_dim


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
    torch.Tensor of same shape as y
        The CRPS for each sample.
    """
    normal_dist = Normal(0, 1)
    std_y = (y - mu) / sigma
    crps = sigma * (std_y * (2 * normal_dist.cdf(std_y) - 1)
                    + 2 * torch.exp(normal_dist.log_prob(std_y))
                    - 1 / torch.sqrt(torch.tensor(torch.pi)))
    return crps


def normal_mae(predicted_params, y, reduce_mean=True):
    """
    Returns the MAE between the predicted mean (which is also the median) and the true value.
    """
    # Flatten the tensors to get rid of the time dimension
    predicted_params = predicted_params.view(-1, 2)
    y = y.view(-1)
    mu = predicted_params[:, 0]
    mae = torch.abs(mu - y)
    if reduce_mean:
        return mae.mean()
    else:
        return mae


def normal_coverage(predicted_params, y, alpha=0.0101, reduce_mean=True):
    """
    Computes the coverage of the predicted intervals at the alpha level.
    The coverage is the proportion of true values that fall within the predicted intervals.
    
    Parameters
    ----------
    predicted_params : torch.Tensor of shape (N, T, 2)
        The predicted parameters of the normal distribution for each sample and time step.
    y : torch.Tensor of shape (N, T)
        The true values for each sample and time step.
    alpha : float
        The level of the interval considered: I = [F^-1(alpha/2), F^-1(1-alpha/2)].
        The default corresponds to 2 * 0.5 / 99,
        which is the same level used in the QuantilesComposite distribution.
    reduce_mean : bool
        Whether to reduce the mean over the batch.

    Returns
    -------
    torch.Tensor of shape (1,)
        The coverage at the alpha level.
    """
    mu, sigma = predicted_params[:, :, 0], predicted_params[:, :, 1]
    normal_dist = Normal(mu, sigma)
    lower_bound = normal_dist.icdf(torch.tensor(alpha / 2))
    upper_bound = normal_dist.icdf(torch.tensor(1 - alpha / 2))
    coverage = ((y >= lower_bound) & (y < upper_bound)).float()
    # Reduce over the time dimension
    coverage = coverage.mean(dim=1)
    if reduce_mean:
        return coverage.mean()
    else:
        return coverage


class NormalDistribution:
    """
    Object that contains the loss function, cdf and metrics for normal distributions.

    Parameters
    ----------
    """

    def __init__(self):
        self.n_parameters = 2
        self.is_multivariate = False

        # Metrics
        # The MAE is the mean absolute error between the predicted mean (which is also the
        # median) and the true value.
        self.metrics = {
            'MAE': normal_mae,
            'CRPS': self.loss_function,
            'Coverage': normal_coverage
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

    def activation(self, predicted_params):
        """
        Applies the Softplus activation to the predicted standard deviation,
        to ensure that it is positive.

        Parameters
        ----------
        predicted_params : torch.Tensor of shape (N, T, 2)
            The predicted parameters of the normal distribution for each sample and time step.
        """
        mu, sigma = predicted_params[:, :, 0], predicted_params[:, :, 1]
        return torch.stack([mu, torch.nn.functional.softplus(sigma)], dim=-1)

    def loss_function(self, predicted_params, y, reduce_mean=True):
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
        reduce_mean : bool
            Whether to reduce the mean of the loss over the batch.

        Returns
        -------
        If reduce_mean is True, returns a torch.Tensor of shape (1,)
        Otherwise, returns a torch.Tensor of shape (N,)
        """
        mu, sigma = predicted_params[:, :, 0], predicted_params[:, :, 1]
        # Compute the CRPS
        loss = normal_crps(mu, sigma, y)
        # Compute the mean over the time steps
        loss = loss.mean(dim=-1)
        # If necessary, reduce to the mean over the batch
        if reduce_mean:
            return loss.mean()
        else:
            return loss

    def denormalize(self, predicted_params, task, dataset):
        """
        De-normalizes the predicted parameters of the distribution.

        Parameters
        ----------
        predicted_params : torch.Tensor of shape (N, T, 2)
            The predicted parameters of the normal distribution for each sample and time step.
        task : str
            The name of the task.
        dataset : dataset, as an object that implements the
            get_normalization_constants method.

        Returns
        -------
        torch.Tensor of shape (N, T, 2)
            The de-normalized parameters.
        """
        # Get the normalization constants from the tasks dictionary
        means, stds = dataset.get_normalization_constants(task)
        # The normalization constants are of shape (T,) but the predicted means and stds
        # are of shape (N, T), so we need to add a new dimension to the constants.
        means = means.unsqueeze(0).to(predicted_params.device)
        stds = stds.unsqueeze(0).to(predicted_params.device)
        # De-normalize the predictions by rescaling the predicted mean and std
        # Mean of the distribution:
        predicted_params[:, :, 0] = predicted_params[:, :, 0] * stds + means
        # Standard deviation of the distribution:
        predicted_params[:, :, 1] = predicted_params[:, :, 1] * stds
        return predicted_params

    def pdf(self, predicted_params, x):
        """
        Computes the probability density function of the distribution.

        Parameters
        ----------
        pred: torch.Tensor of shape (N, T, 2) or (T, 2)
            The predicted parameters of the normal distribution for each sample and time step.
        x : torch.Tensor of shape (N, T) or (T,)
            The values at which to compute the probability density function.

        Returns
        -------
        pdf : torch.Tensor of shape (N, T) or (T,)
            The probability density function of the distribution.
        """
        # y -> (N, T), y_pred -> (N, T, 1)
        x, predicted_params = add_batch_dim(x, predicted_params)
        mu, sigma = predicted_params[:, :, 0], predicted_params[:, :, 1]
        normal_dist = Normal(mu, sigma)
        return normal_dist.log_prob(x).exp()

    def hyperparameters(self):
        return {}
