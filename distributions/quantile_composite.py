"""
Defines the QuantileCompositeDistribution class.
"""
import torch
from loss_functions.quantiles import CompositeQuantileLoss, QuantilesCRPS
from utils.utils import add_batch_dim


class QuantileCompositeDistribution:
    """
    Object that can define a distribution from a set of quantiles.

    Parameters
    ----------
    """
    def __init__(self):
        self.probas = torch.linspace(0.01, 0.99, 99)
        self.n_parameters = len(self.probas)
        self.is_multivariate = False

        # Define the loss function
        self.loss_function = CompositeQuantileLoss(self.probas)

        # Define the metrics
        self.metrics = {}
        # Then, the MAE (corresponding to the 0.5 quantile)
        self.metrics["MAE"] = CompositeQuantileLoss(torch.tensor([0.5]))
        # Then, the CRPS
        self.metrics["CRPS"] = QuantilesCRPS(self.probas)

    def activation(self, predicted_params):
        # Identity activation
        return predicted_params

    def denormalize(self, predicted_params, task, dataset):
        """
        Denormalizes the predicted values.

        Parameters
        ----------
        predicted_params : torch.Tensor of shape (N, T, Q)
            The predicted values for each sample and time step.
        task : str
        dataset: dataset object that implements the get_normalization_constants method.

        Returns
        -------
        torch.Tensor of shape (N, T, Q)
            The denormalized predicted quantiles.
        """
        # Retrieve the normalization constants, of shape (T,)
        means, stds = dataset.get_normalization_constants(task)
        # Reshape the means and stds to be broadcastable and move them to the same device
        # as the predictions
        means = means.view(1, -1, 1).to(predicted_params.device)
        stds = stds.view(1, -1, 1).to(predicted_params.device)
        # De-normalize the predictions
        return predicted_params * stds + means

    def _preprocess_input(self, predicted_params, x):
        """
        Internal method to preprocess the input of the pdf and cdf methods.
        """
        # x -> (N, T) and predicted_params -> (N, T, Q)
        x, predicted_params = add_batch_dim(x, predicted_params)
        # First, we need to find for each sample in which bin the true value falls
        # (i.e. which two quantiles it is between)
        # For every n we want i such that pq[n, i-1] <= y[n, 0] < pq[n, i]
        idx = torch.searchsorted(predicted_params,
                                 x.unsqueeze(2), side="right")
        return predicted_params, x, idx

    def pdf(self, predicted_params, x):
        """
        Computes the probability density function of the distribution.

        Parameters
        ----------
        predicted_params : torch.Tensor of shape (N, T, Q) or (T, Q)
            The predicted quantiles of the distribution.
        x : torch.Tensor of shape (N, T) or (T,)
            The values at which to compute the probability density function.

        Returns
        -------
        pdf : torch.Tensor of shape (N, T) or (T,)
            The probability density function of the distribution.
        """
        pq, x, idx = self._preprocess_input(predicted_params, x)
        N, T = x.shape
        # The pdf is defined as 0 if x < q1 or x >= qK;
        # For q1 <= x < qK, the pdf is (tau_i - tau_i-1) / (pq_i - pq_i-1)
        # where i is such that pq_i-1 <= x < pq_i
        in_mask = (x >= pq[:, :, 0]) & (x < pq[:, :, -1])
        # We'll first compute the pdf as if every x was within the bounds
        # To do so we need idx to be in [1, K-1]
        idx = idx.clamp(1, self.n_parameters - 1)
        dpq = pq = pq.gather(2, idx) - pq.gather(2, idx - 1)
        dt = self.probas[1:] - self.probas[:-1]
        res = (dt[idx - 1] / dpq)
        # Now we set the pdf to 0 for the values of x that are outside the bounds
        res[~in_mask] = 0
        # If x is of shape (1, T), remove the batch dimension
        if N == 1:
            res = res.squeeze(0)
        return res

    def cdf(self, predicted_params, x):
        """
        Computes the cumulative distribution function of the distribution.

        Parameters
        ----------
        predicted_params : torch.Tensor of shape (N, T, Q) or (T, Q)
            The predicted quantiles of the distribution.
        x : torch.Tensor of shape (N, T) or (T,)
            The values at which to compute the cumulative distribution function.

        Returns
        -------
        cdf : torch.Tensor of shape (N, T) or (T,)
            The cumulative distribution function of the distribution.
        """
        pq, x, idx = self._preprocess_input(predicted_params, x)
        N, T = x.shape
        # The cdf is defined as 0 if x < q1;
        # For q1 <= x < qK, the cdf is tau_i-1 where i is such that pq_i-1 <= x < pq_i
        # Finally, the cdf is 1 if x >= qK
        res = torch.zeros_like(x)
        in_mask = x >= pq[:, 0]
        idx_in = idx[in_mask]
        res[in_mask] = self.probas[idx_in - 1]
        # Reshape back to the original shape
        res = res.view(N, T)
        # If x is of shape (1, T), remove the batch dimension
        if N == 1:
            res = res.squeeze(0)
        return res

    def hyperparameters(self):
        """
        Returns the hyperparameters of the distribution. Here, it is
        the minimum and maximum values of the distribution, as well as
        the quantiles defining the distribution.

        Returns
        -------
        hyperparameters : dict
            The hyperparameters of the distribution.
        """
        return {"min_value": self.min_value,
                "max_value": self.max_value,
                "probas": self.probas}

