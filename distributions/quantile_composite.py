"""
Defines the QuantileCompositeDistribution class.
"""

import torch
import torch.nn.functional as F
from loss_functions.quantiles import CompositeQuantileLoss, QuantilesCRPS, quantiles_coverage
from loss_functions.common import CoveredCrps
from utils.utils import add_batch_dim


class QuantileCompositeDistribution:
    """
    Object that can define a distribution from a set of quantiles.
    Remark: to ensure that the quantiles do not cross, the predictions
    should be the first quantile q0, followed by the successive differences
    between the quantiles, i.e. q1 - q0, q2 - q1, ..., qQ - qQ-1.
    A Sotfplus activation is applied to the differences to ensure that they
    are positive.

    Parameters
    ----------
    Q: int, optional
        The number of quantiles predicted. Defaults to 99.
    """

    def __init__(self, Q=99):
        # We use the optimal quantiles, i.e. those which best optimize the CRPS.
        # See (Zamo et Naveau, 2018) for more details.
        self.probas = torch.linspace(1 / Q, 1, Q) - 0.5 / Q
        self.n_parameters = len(self.probas)
        self.is_multivariate = False

        # Define the loss function
        self.loss_function = CompositeQuantileLoss(self.probas)

        # Define the metrics
        self.metrics = {}
        # The "MAE" is defined here as the L1 norm between y and the 0.5-quantile
        self.metrics["MAE"] = CompositeQuantileLoss(torch.tensor([0.5]))
        crps_fn = QuantilesCRPS(self.probas)
        self.metrics["CRPS"] = crps_fn
        self.metrics["Coverage"] = quantiles_coverage
        self.metrics["Covered CRPS"] = CoveredCrps(crps_fn, quantiles_coverage, 1.0)

    def activation(self, predicted_params):
        """
        Converts the parameters predicted by the model to quantiles.

        Parameters
        ----------
        predicted_params : torch.Tensor of shape (N, T, Q)
            The predicted parameters. For a given sample n and time step t,
            predicted_params[n, t, 0] is interpreted as the first quantile,
            while predicted_params[n, t, k] = q_k - q_k-1 for k > 0.

        Returns
        -------
        quantiles : torch.Tensor of shape (N, T, Q)
            The predicted quantiles.
        """
        # Apply the softplus activation to the differences to ensure that they are positive
        pred = torch.cat(
            [predicted_params[:, :, :1], F.softplus(predicted_params[:, :, 1:])], dim=2
        )
        # Compute the quantiles
        return torch.cumsum(pred, dim=2)

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
        idx = torch.searchsorted(predicted_params, x.unsqueeze(2), side="right")
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
        # To do so we need idx to be in [1, K-1] to avoid out of bounds errors
        idx = idx.clamp(1, self.n_parameters - 1)
        dpq = pq.gather(2, idx) - pq.gather(2, idx - 1)
        dt = self.probas[1:] - self.probas[:-1]
        res = dt[idx - 1] / dpq
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
        # For q1 <= x < qK, the cdf is tau_i-1 + (x - pq_i-1) / (pq_i - pq_i-1)
        # where i is such that pq_i-1 <= x < pq_i
        # Finally, the cdf is 1 if x >= qK
        res = torch.zeros_like(x)
        # We'll first compute the pdf as if every x was within the bounds
        # To do so we need idx to be in [1, K-1] to avoid out of bounds errors
        idx = idx.clamp(1, self.n_parameters - 1)
        pql = pq.gather(2, idx - 1)  # pq_i-1
        dpq = pq.gather(2, idx) - pql  # pq_i - pq_i-1
        dt = self.probas[1:] - self.probas[:-1]
        slope = dt[idx - 1] / dpq
        res = self.probas[idx - 1] + slope * (x.unsqueeze(-1) - pql)
        # Now, set the cdf to 0 or 1 for the values of x that are outside the bounds
        res[x < pq[:, :, 0]] = 0
        res[x >= pq[:, :, -1]] = 1
        # Reshape back to the original shape
        res = res.view(N, T)
        # If x is of shape (1, T), remove the batch dimension
        if N == 1:
            res = res.squeeze(0)
        return res

    def hyperparameters(self):
        """
        Returns
        -------
        hyperparameters : dict
            The hyperparameters of the distribution.
        """
        return {"probas": self.probas}
