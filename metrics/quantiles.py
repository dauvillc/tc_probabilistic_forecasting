"""
Implements various functions to evaluate the quality of a set of
predicted quantiles.
"""
import torch
import numpy as np
from utils.utils import to_numpy


class Quantiles_eCDF:
    """
    Given a set of predicted quantiles, computes the empirical CDF
    at any point y.

    Parameters
    ----------
    quantiles: array-like of floats
        The quantiles that define the empirical CDF, between 0 and 1.
    min_val: float
        Minimum value of the empirical CDF (beginning of the support).
    max_val: float
        Maximum value of the empirical CDF (end of the support).
    """
    def __init__(self, quantiles, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        # Convert the quantiles to a torch.Tensor
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)
        # Add 0 and 1 to the quantiles for generality
        self.quantiles = torch.cat((torch.tensor([0]), self.quantiles, torch.tensor([1])))

    def __call__(self, predicted_quantiles, y):
        """
        Computes the empirical CDF from the predicted quantiles,
        and then evaluates it at y.

        Parameters
        ----------
        predicted_quantiles: torch.Tensor of shape (N, Q)
            where N is the number of samples and Q is the number of quantiles.
            The predicted quantiles.
        y: torch.Tensor of shape (N,)
            The values at which the empirical CDF is evaluated.

        Returns
        -------
        The empirical CDF at y, as a torch.Tensor.
        """
        # Add the maximum value to the predicted quantiles for generality
        predicted_quantiles = torch.cat((predicted_quantiles,
                                         torch.full_like(predicted_quantiles[:, :1], self.max_val)), dim=1)
        # Find the index of the predicted quantile that is just below y
        # (or equal to y)
        index = torch.searchsorted(self.quantiles, y, right=False) - 1
        # Compute the empirical CDF at y as the corresponding quantile
        return predicted_quantiles[:, index]



class Quantiles_inverse_eCDF:
    """
    Given a set of predicted quantiles, computes the inverse of the
    empirical CDF at any point u.

    Parameters
    ----------
    quantiles: array-like of floats
        The quantiles that define the empirical CDF, between 0 and 1.
    min_val: float
        Minimum value of the empirical CDF (beginning of the support).
    max_val: float
        Maximum value of the empirical CDF (end of the support).
    """
    def __init__(self, quantiles, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        # Convert the quantiles to a torch.Tensor
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)
        # Add 0 and 1 to the quantiles for generality
        self.quantiles = torch.cat((torch.tensor([0]), self.quantiles, torch.tensor([1])))

    def __call__(self, predicted_quantiles, u):
        """
        Computes the inverse of the empirical CDF from the predicted quantiles,
        and then evaluates it at u.

        Parameters
        ----------
        predicted_quantiles: torch.Tensor of shape (N, Q)
            where N is the number of samples and Q is the number of quantiles.
            The predicted quantiles.
        u: float 
            The probability at which the inverse empirical CDF is computed.

        Returns
        -------
        The inverse empirical CDF at u, as a torch.Tensor.
        """
        # Add the minimum value to the predicted quantiles for generality
        predicted_quantiles = torch.cat((torch.full_like(predicted_quantiles[:, :1], self.min_val),
                                            predicted_quantiles), dim=1)
        # Find the index of the quantile that is just below u 
        index = torch.searchsorted(self.quantiles, u, right=True) - 1
        # Compute the inverse empirical CDF at u as the corresponding quantile
        return predicted_quantiles[:, index]


class QuantilesCRPS:
    """
    Computes the CRPS for a set of predicted quantiles.

    Parameters
    ----------
    quantiles: array-like or torch.Tensor
        The quantiles that define the empirical CDF, between 0 and 1.
    min_val: float
        Minimum value of the empirical CDF (beginning of the support).
    max_val: float
        Maximum value of the empirical CDF (end of the support).
    """
    def __init__(self, quantiles, min_val, max_val):
        self.quantiles = to_numpy(quantiles)
        self.min_val = min_val
        self.max_val = max_val
        # Add 0 and 1 to the quantiles for generality
        self.quantiles = np.concatenate(([0], self.quantiles, [1]))

    def __call__(self, predicted_quantiles, y):
        """
        Computes the CRPS from the predicted quantiles.

        Parameters
        ----------
        predicted_quantiles: array-like or torch.Tensor, of shape (N, T, Q)
            where N is the number of samples, T is the number of time steps
            and Q is the number of quantiles.
            The predicted quantiles.
        y: array-like or torch.Tensor, of shape (N, T)
            The true values.

        Returns
        -------
        The average CRPS over all samples and time steps, as a float.
        """
        predicted_quantiles = to_numpy(predicted_quantiles)
        y = to_numpy(y)
        # Reshape the predicted quantiles and the true values to
        # (N * T, Q) and (N * T,) respectively
        predicted_quantiles = predicted_quantiles.reshape(-1, predicted_quantiles.shape[-1])
        y = y.reshape(-1)
        # As the searchsorted method does not work for 2D arrays,
        # we loop over the samples and time steps
        crps = []
        for i in range(predicted_quantiles.shape[0]):
            # Add the minimum and maximum values to the predicted quantiles
            # for generality
            pred_i = np.concatenate(([self.min_val], predicted_quantiles[i], [self.max_val]))
            # Find the index of the predicted quantile that is just below y
            index = np.searchsorted(pred_i, y[i], side='right') - 1
            # Compute the area under the empirical CDF before the index
            if index == 0:
                area = 0
            else:
                area = np.sum((pred_i[2:index + 1] - pred_i[1:index]) * self.quantiles[1:index] ** 2)
            # Compute the area under the empirical CDF between the index and
            # index + 1, where the observed value lies
            area += (y[i] - pred_i[index]) * self.quantiles[index] ** 2
            area += (pred_i[index + 1] - y[i]) * (1 - self.quantiles[index]) ** 2
            # Compute the area under the empirical CDF after the index + 1
            area += np.sum((pred_i[index + 2:] - pred_i[index + 1:-1]) * (1 - self.quantiles[index + 1:-1]) ** 2)
            crps.append(area)
        return np.mean(crps)
