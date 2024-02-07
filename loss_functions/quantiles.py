"""
Implements various functions to evaluate the quality of a set of
predicted quantiles.
"""
import torch
import torch.nn as nn


class CompositeQuantileLoss(nn.Module):
    """
    Implements the multiple quantile loss function.
    The prediction should have the shape (N, T, Q) where N is the batch size,
    T is the number of time steps, and Q is the number of quantiles.

    Parameters
    ----------
    probas: tensor of shape (Q,)
        The probabilities associated with the quantiles to estimate.
    """
    def __init__(self, probas):
        super().__init__()
        self.probas = probas
        # The term (y_true - y_pred) will have shape (N, T, Q).
        self.probas = self.probas.unsqueeze(0)  # (1, Q)

    def __call__(self, y_pred, y_true, reduce_mean=True): 
        # Transfer the probas to the device of y_pred 
        probas = self.probas.to(y_pred.device)
        # y_pred has shape (N, T, Q), while y_true has shape (N, T)
        y_true = y_true.unsqueeze(2)  # (N, T, 1)
        # Compute the quantile loss for each sample and each quantile
        loss = torch.max(probas * (y_true - y_pred),
                         (1 - probas) * (y_pred - y_true))
        # Reduce the loss over the quantiles
        loss = loss.mean(dim=(1, 2))
        # Reduce the loss over the batch
        if reduce_mean:
            loss = loss.mean()
        return loss

class Quantiles_eCDF:
    """
    Given a set of predicted quantiles, computes the empirical CDF
    at any point y.

    Parameters
    ----------
    quantiles: torch.Tensor of shape (Q,)
        The quantiles that define the empirical CDF, between 0 and 1.
    min_val: float
        Minimum value of the empirical CDF (beginning of the support).
    max_val: float
        Maximum value of the empirical CDF (end of the support).
    """
    def __init__(self, quantiles, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.quantiles = quantiles
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
    quantiles: torch Tensor of shape (Q,)
        The quantiles that define the empirical CDF, between 0 and 1.
    min_val: float
        Minimum value of the empirical CDF (beginning of the support).
    max_val: float
        Maximum value of the empirical CDF (end of the support).
    """
    def __init__(self, quantiles, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.quantiles = quantiles
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
    probas: torch.Tensor
        The probabilities that define the predicted quantiles, between 0 and 1.
    """
    def __init__(self, probas):
        self.probas = probas
        # Compute the differences between successive probabilities (delta tau)
        self.dt = self.probas[1:] - self.probas[:-1]

    def __call__(self, predicted_quantiles, y, reduce_mean=True):
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
        reduce_mean: bool, optional
            Whether to reduce the CRPS over the batch and time steps.

        Returns
        -------
        The average CRPS over all samples and time steps, as a float.
        """
        # Move dt to the device of predicted_quantiles
        dt = self.dt.to(predicted_quantiles.device)
        # Move the probas to the device of y
        tau = self.probas.to(y.device)
        # Flatten the predicted quantiles and the true values to get rid of the time dimension
        pq = predicted_quantiles.view(-1, predicted_quantiles.shape[2])  # (N * T, Q)
        y = y.view(-1)  # (N * T,)
        # Compute the CRPS, considering a linear interpolation between the predicted quantiles
        # First, we need to find for each sample in which bin the true value falls
        # (i.e. which two quantiles it is between)
        # For every n we want i such that pq[n, i-1] <= y[n, 0] < pq[n, i]
        idx = torch.searchsorted(pq, y.unsqueeze(1), side="right").squeeze(1)
        # Compute the differences between the predicted quantiles (delta pq)
        dpq = pq[:, 1:] - pq[:, :-1]
        res = torch.zeros(y.shape[0], device=y.device)
        # Compute the integral of F**2 before the true value, considering trapezoidal integration
        ts = tau ** 2
        it = (1 - tau)
        its = it ** 2
        dts = dt ** 2
        n_quantiles = pq.shape[1]
        # First, we'll compute the integral over every bin strictly before and after the true value
        for i in range(n_quantiles - 1):
            # Compute the integral of F**2 from pq[n, i] to pq[n, i+1]
            term = (ts[i] + tau[i] * dt[i] + dts[i] / 3) * dpq[:, i]
            # This value is only valid the quantile just below y is at least i+1
            term[idx - 1 < i + 1] = 0
            res += term
        # After the true value
        for i in range(n_quantiles - 1):
            # Compute the integral of F**2 from pq[n, i] to pq[n, i+1]
            term = (its[i] + it[i] * dt[i] + dts[i] / 3) * dpq[:, i]
            # This value is only valid when the quantile just above y is at most i
            term[idx > i] = 0
            res += term
        # Now, there are three cases:
        # * The observation is outside lower than the lowest quantile
        # * The observation is within the predicted distribution
        # * The observation is outside higher than the highest quantile
        
        # We'll first treat the case where the observation is within the predicted quantiles
        # The following is only valid when idx == 0 or idx == n_quantiles (non-extreme indexes)
        nem = (idx > 0) & (idx < n_quantiles)  # Which sample have y within the prediction
        nei = idx[nem]  # For those samples, which quantile is just above y
        # Linearly interpolate the value of F at y[n, 0].
        # Note: nei is the index of the quantile just above y
        tau_y = (tau[nei] - tau[nei - 1]) / dpq[nem, nei - 1] * (y[nem] - pq[nem, nei - 1]) + tau[nei - 1]
        dt_y = tau_y - tau[nei - 1]
        # Compute the integral between pq[n, nei-1] and y[n, 0]
        term = (ts[nei - 1] + tau[nei - 1] * dt_y + dt_y ** 2 / 3) * (y[nem] - pq[nem, nei - 1])
        # Compute the integral between y[n, 0] and pq[n, nei]
        dt_y = tau[nei] - tau_y
        term += ((1 - tau_y) ** 2 + (1 - tau_y) * dt_y + dt_y ** 2 / 3) * (pq[nem, nei] - y[nem])
        res[nem] += term

        # Now, when the observation is outside the lowest quantile
        zero_idx = idx == 0
        res[zero_idx] += pq[zero_idx, 0] - y[zero_idx]
        # When the observation is outside the highest quantile
        max_idx = idx == n_quantiles
        res[max_idx] += (1 - tau[-1]) * (y[max_idx] - pq[max_idx, -1])

        if reduce_mean:
            res = res.mean()
        return res
