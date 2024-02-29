"""
Tests the implementation of the quantile CRPS.
"""
import sys
sys.path.append("./")
from loss_functions.quantiles import QuantilesCRPS
from distributions.normal import normal_crps
from torch.distributions import Normal
import torch


def nrg_crps(x, y):
    """
    Estimates the CRPS via the NRG method.
    (See Naveau et Zamo 2018)

    Parameters
    ----------
    x: torch.Tensor of shape (m,)
        Quantiles of the predicted distribution.
    y: torch.Tensor of shape (1,)
        The observation.
    """
    m = x.shape[0]
    crps = torch.abs(x - y).mean()
    crps -= (0.5 / m ** 2) * \
        torch.abs(x.unsqueeze(1) - x).sum(dim=0).sum()
    return crps


def fair_crps(x, y):
    """
    Estimates the CRPS via the Fair (PWM) method.

    Parameters
    ----------
    x : torch.Tensor of shape (m,)
        The samples from the predictive distribution.
    y : torch.Tensor of shape (,)
        The observation.
    """
    m = x.shape[0]
    crps = torch.abs(x - y).mean()
    crps -= (0.5 / (m * (m - 1))) * \
        torch.abs(x.unsqueeze(1) - x).sum(dim=0).sum()
    return crps


def quantiles_integral_crps(tau, pred_quantiles, y):
    """
    Estimates the CRPS as the integral of the quantile loss from 0 to 1.
    This requires a large number of quantiles to be accurate.
    (See Berrisch et al. 2021, "CRPS Learning").
    Only implemented for equidistant quantiles.

    Parameters
    ----------
    tau: torch.Tensor of shape (Q,)
        Probabilities of the quantiles.
    pred_quantiles : torch.Tensor of shape (Q,)
    y: torch.Tensor of shape (1,)
    """
    # Compute the quantile loss for each sample and each quantile
    loss = torch.max(tau * (y - pred_quantiles),
                     (1 - tau) * (pred_quantiles - y))
    # Compute the integral of the quantile loss
    return 2 * loss.sum() / tau.shape[0]


if __name__ == "__main__":
    # Test 1 - Two quantiles are predicted only
    y = torch.tensor([2])
    pred = torch.tensor([[[1, 3]]])
    probas = torch.tensor([0.25, 0.75])
    crps = QuantilesCRPS(probas)(pred, y)
    print("CRPS for test 1:", crps)

    # Test 2 - Two quantiles, observation is lower than the lowest quantile
    y = torch.tensor([0])
    pred = torch.tensor([[[1, 3]]])
    probas = torch.tensor([0.25, 0.75])
    crps = QuantilesCRPS(probas)(pred, y)
    print("CRPS for test 2:", crps)

    # Test 3 - Two quantiles, observation is higher than the highest quantile
    y = torch.tensor([4])
    pred = torch.tensor([[[1, 3]]])
    probas = torch.tensor([0.25, 0.75])
    crps = QuantilesCRPS(probas)(pred, y)
    print("CRPS for test 3:", crps)

    # Test 4 - Two quantiles, observation is exactly the second quantile
    y = torch.tensor([3])
    pred = torch.tensor([[[1, 3]]])
    probas = torch.tensor([0.25, 0.75])
    crps = QuantilesCRPS(probas)(pred, y)
    print("CRPS for test 4:", crps)

    # Test 5 - Compute the CRPS for a normal distribution in different ways:
    # 1. By sampling i.i.d. samples and using the fair form of the CRPS;
    # 2. By using the closed-form formula for the CRPS of a normal distribution;
    # 3. By using the quantile CRPS.
    # 4. By using the integral of the quantile loss from 0 to 1.
    # 5. By using the INT / NRG form of the CRPS, considering the quantiles
    #    as non-exchangeable members of an ensemble.
    mu = torch.tensor([0.])
    sigma = torch.tensor([1.])
    y = torch.tensor([4])
    dist = Normal(mu, sigma)
    tau = torch.linspace(0.0001, 0.9999, 9999)
    # Method 1
    samples = dist.sample((20000,))
    crps_fair = fair_crps(samples, y)
    print("CRPS for test 5 (sampling method):", crps_fair)
    # Method 2
    crps_closed_form = normal_crps(mu, sigma, y)
    print("CRPS for test 5 (closed-form method):", crps_closed_form)
    # Method 3
    quantiles = dist.icdf(tau).view(1, 1, -1)
    crps_quantiles = QuantilesCRPS(tau)(quantiles, y)
    print("CRPS for test 5 (quantile method):", crps_quantiles)
    # Method 4
    crps_ql_integral = quantiles_integral_crps(tau, quantiles.view(-1), y)
    print("CRPS for test 5 (quantile integral method):", crps_ql_integral)
    # Method 5
    crps_nrg = nrg_crps(quantiles.view(-1), y)
    print("CRPS for test 5 (nrg method):", crps_nrg)
