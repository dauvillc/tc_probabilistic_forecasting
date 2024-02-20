"""
Implements the DeterministicDistribution class, which is useful to integrate a deterministic
model into the probabilistic pipeline.
"""
import torch
from utils.utils import add_batch_dim


def mse(y_pred, y_true, reduce_mean=True):
    """
    MSE Loss.

    Parameters
    ----------
    y_pred : torch.Tensor of shape (N, T, 1)
        The predicted values for each sample and time step.
    y_true : torch.Tensor of shape (N, T)
    """
    y_pred = y_pred.squeeze(-1)
    loss = (y_pred - y_true) ** 2
    # Average over the time steps
    loss = torch.mean(loss, dim=1)
    # If reduce_mean is True, average over the batch
    if reduce_mean:
        return torch.mean(loss)
    else:
        return loss


def mae(y_pred, y_true, reduce_mean=True):
    """
    Flattens the tensor and computes the MAE.
    """
    y_pred = y_pred.squeeze(-1)
    loss = torch.abs(y_pred - y_true)
    # Average over the time steps
    loss = torch.mean(loss, dim=1)
    # If reduce_mean is True, average over the batch
    if reduce_mean:
        return torch.mean(loss)
    else:
        return loss


def flatten_RMSE(y_pred, y_true, reduce_mean=True):
    """
    Flattens the tensor and computes the RMSE.
    """
    return torch.sqrt(torch.mean((y_pred.flatten() - y_true.flatten()) ** 2))


class DeterministicDistribution:
    """
    Object that defines a deterministic distribution, i.e. the CDF is the step function
    that assigns probability 1 after the predicted value, and 0 before it.

    Parameters
    ----------
    """
    def __init__(self):
        # The distribution P(y|x) is deterministic, so it is characterized by a single
        # parameter, which is the predicted value.
        self.n_parameters = 1
        self.is_multivariate = False

        # The output of the model will have shape (N, T, V) where N is the batch size,
        # T is the number of time steps and V is the number of output variables.
        self.loss_function = mse

        # Define the metrics
        self.metrics = {
                'RMSE': flatten_RMSE,
                'MAE': mae,
                'CRPS': mae  # The CRPS is the MAE for a deterministic distribution
                }

    def hyperparameters(self):
        """
        Returns the hyperparameters of the distribution.
        """
        return {}

    def pdf(self, pred, x):
        """
        Computes the probability density function of the distribution.

        Parameters
        ----------
        pred: torch.Tensor of shape (N, T, 1) or (T, 1)
            Predictes values for each sample and time step.
        x : torch.Tensor of shape (N, T) or (T,)
            The values at which to compute the probability density function.

        Returns
        -------
        pdf : torch.Tensor of shape (N, T) or (T,)
            The probability density function of the distribution.
        """
        # y -> (N, T), y_pred -> (N, T, 1)
        x, pred = add_batch_dim(x, pred)
        pred = pred.squeeze(-1)  # (N, T)
        # The real pdf would be zero everywhere.
        # We'll simulate a Dirac delta function, which is 0 everywhere except at the
        # predicted value. We'll indicate a value of 1 in a small neighborhood around the
        # predicted value.
        res = torch.zeros_like(x)
        res[torch.abs(x - pred)/pred <= 2e-2] = 1
        return res

    def cdf(self, y_pred, y):
        """
        Given a deterministic distribution that always outputs y, the CDF is defined as
        1 for y_pred >= y, and 0 otherwise.
        """
        return torch.heaviside(y - y_pred, torch.tensor([1.0]))

    def inverse_cdf(self, y, u):
        """
        Given a deterministic distribution that always outputs y, the inverse CDF is
        defined as y for any u in [0, 1].
        """
        return y

    def activation(self, predicted_params):
        # Identity activation function
        return predicted_params

    def denormalize(self, predicted_params, task, dataset):
        """
        Denormalizes the predicted values.

        Parameters
        ----------
        predicted_params : torch.Tensor of shape (N, T, V)
            The predicted values for each sample and time step.
        task : str
        dataset : dataset, as an object that implements the
            get_normalization_constants method.

        Returns
        -------
        torch.Tensor of shape (N, T, V)
            The denormalized predicted values.
        """
        # Retrieve the normalization constants, of shape (T * V)
        means, stds = dataset.get_normalization_constants(task)
        # Reshape the means and stds to be broadcastable and move them to the same device
        # as the predictions
        means = means.view(predicted_params.shape[1:]).to(predicted_params.device)
        stds = stds.view(predicted_params.shape[1:]).to(predicted_params.device)
        # De-normalize the predictions
        return predicted_params * stds + means
