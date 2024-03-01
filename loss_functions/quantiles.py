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
        """
        Parameters
        ----------
        y_pred: torch.Tensor, of shape (N, T, Q)
            The predicted quantiles.
        y_true: torch.Tensor, of shape (N, T)
            The true values.
        reduce_mean: bool, optional
            Whether to reduce the loss over the batch dimension.
        """
        # Transfer the probas to the device of y_pred
        probas = self.probas.to(y_pred.device)
        # y_pred has shape (N, T, Q), while y_true has shape (N, T)
        y_true = y_true.unsqueeze(2)  # (N, T, 1)
        # Compute the quantile loss for each sample and each quantile
        loss = torch.max(probas * (y_true - y_pred), (1 - probas) * (y_pred - y_true))
        # Reduce the loss over the quantiles
        loss = loss.mean(dim=(1, 2))
        # Reduce the loss over the batch
        if reduce_mean:
            loss = loss.mean()
        return loss


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
        N, T, Q = predicted_quantiles.shape
        # Move dt to the device of predicted_quantiles
        dt = self.dt.to(predicted_quantiles.device)
        # Move the probas to the device of y
        tau = self.probas.to(y.device)
        # Flatten the predicted quantiles and the true values to get rid of the time dimension
        # (N * T, Q)
        pq = predicted_quantiles.view(-1, predicted_quantiles.shape[2])
        y = y.view(-1, 1)  # (N * T, 1)
        # Compute the CRPS, considering a linear interpolation between the predicted quantiles.
        # We'll first compute the integral within each bin of the predicted quantiles.
        # We'll make some pre-computations:
        pql, pqr = (
            pq[:, :-1],
            pq[:, 1:],
        )  # Left and right predicted quantiles (p_{k} and p_{k+1})
        t = tau[:-1]  # Tau without the last probability
        dpq = pqr - pql  # Differences between successive predicted quantiles
        a = dt / dpq  # Slopes of the linear interpolation
        dpqs = dpq ** 2  # (p_{k+1} - p_{k})^2
        dpqc = dpqs * dpq  # (p_{k+1} - p_{k})^3
        v = t - (y < pql).float()  # t - 1 if y < pq_k, t if y >= pq_k
        # Compute the integral within each bin
        # Note: if y is within a bin, the integral within that bin will have to be
        # computed differently after.
        integral = (
            dpq * v ** 2 + v * a * dpqs + a ** 2 * dpqc / 3
        )
        # There are now three cases:
        # 1. y is between two quantiles q_{k0} <= y < q_{k0 + 1}
        # 2. y is below the first quantile y < q_{0}
        # 3. y is above the last quantile y >= q_{Q - 1}
        # We can now the case for each sample of the batch in a vectorized way,
        # using searchsorted:
        # We obtain the index k0 such that pq[n, k0] <= y[n - 1] < pq[n, k0]
        k0 = torch.searchsorted(pq, y, right=True).squeeze(1)
        # For the following, we need y to have shape (N * T,)
        y = y.squeeze(1)
        # Case 1. y is between two quantiles (i.e. k0 > 0 and k0 < Q)
        mask = (k0 > 0) & (k0 < Q)
        if mask.sum() > 0:  # If there are no samples in this case, skip it.
            k0_m, y_m = k0[mask] - 1, y[mask]
            a_m, t_m = a[mask, k0_m], t[k0_m]
            pql_m, pqr_m = pql[mask, k0_m], pqr[mask, k0_m]
            # Pre-computation
            dypl = y_m - pql_m  # y - pq_{k0}
            dypls = dypl ** 2  # (y - pq_{k0})^2
            dyplc = dypls * dypl  # (y - pq_{k0})^3
            dpry = pqr_m - y_m  # pq_{k0 + 1} - y
            dprys = dpry ** 2  # (pq_{k0 + 1} - y)^2
            dpryc = dprys * dpry  # (pq_{k0 + 1} - y)^3
            ty = t_m + a_m * (y_m - pql_m) - 1  # F(y) - 1 for Y within a bin
            where_mask = torch.where(mask)[0]  # integral[mask] would assign to a copy
            # Integral from q_{k0} to y
            integral[where_mask, k0_m] = (
                dypl * t_m ** 2 + t_m * a_m * dypls + a_m ** 2 * dyplc / 3
            )
            # Integral from y to q_{k0 + 1}
            integral[where_mask, k0_m] += (
                dpry * ty ** 2 + ty * a_m * dprys + a_m ** 2 * dpryc / 3
            )
        # Sum the integral over the bins
        integral = integral.sum(dim=1)
        # Case 2. y < q_{0}
        mask = k0 == 0
        if mask.sum() > 0:
            # In this case, we stop considering a linear interpolation.
            # We consider that F(x) = tau_0 from y to q_0
            # integral[mask] += (1 - tau[0]) ** 2 * (pql[mask][:, 0] - y[mask])
            integral[mask] += pql[mask][:, 0] - y[mask]
        # Case 3. y >= q_{Q - 1}
        mask = k0 == Q
        if mask.sum() > 0:
            # In this case, we stop considering a linear interpolation.
            # We consider that F(x) = tau_{Q - 1} from y to q_{Q - 1}
            # integral[mask] += tau[-1] ** 2 * (y[mask] - pqr[mask][:, -1])
            integral[mask] += y[mask] - pqr[mask][:, -1]
        res = integral.view(N, T).sum(dim=1)
        if reduce_mean:
            res = res.mean()
        return res


def quantiles_coverage(pred, y, reduce_mean=True):
    """
    Computes the coverage of the predicted quantiles.
    The coverage is the proportion of true values that fall within the predicted
    quantiles.

    Parameters
    ----------
    pred: torch.Tensor, of shape (N, T, Q)
        The predicted quantiles.
    y: torch.Tensor, of shape (N, T)
        The true values.
    reduce_mean: bool, optional
        Whether to reduce the coverage over the batch.

    Returns
    -------
    The coverage, as a float or a tensor of shape (N,).
    """
    covered = (y >= pred[:, :, 0]) & (y < pred[:, :, -1])
    # Average the coverage over the time steps
    coverage = covered.float().mean(dim=1)
    if reduce_mean:
        coverage = coverage.mean()
    return coverage
