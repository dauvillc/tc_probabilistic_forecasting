"""
Tests the implementation of the quantile CRPS.
"""
import sys
sys.path.append("./")
from loss_functions.quantiles import QuantilesCRPS
from distributions.normal import normal_crps
from argparse import ArgumentParser
from torch.distributions import Normal
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def nrg_crps(x, y):
    """
    Estimates the CRPS via the NRG method.
    (See Naveau et Zamo 2018)

    Parameters
    ----------
    x: torch.Tensor of shape (N, M)
        Quantiles of the predicted distribution.
    y: torch.Tensor of shape (N, 1)
        The observation.
    """
    m = x.shape[1]
    crps = torch.abs(x - y).mean(dim=1)
    x = x.unsqueeze(2)
    crps -= (0.5 / m ** 2) * \
        torch.abs(x.transpose(2, 1) - x).sum(dim=(1, 2))
    return crps


def fair_crps(x, y):
    """
    Estimates the CRPS via the Fair (PWM) method.

    Parameters
    ----------
    x : torch.Tensor of shape (N, M)
        The samples from the predictive distribution.
    y : torch.Tensor of shape (N, 1)
        The observation.
    """
    m = x.shape[1]
    crps = torch.abs(x - y).mean(dim=1)
    x = x.unsqueeze(2)
    crps -= (0.5 / (m * (m - 1))) * \
        torch.abs(x.transpose(2, 1) - x).sum(dim=(1, 2))
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
    pred_quantiles : torch.Tensor of shape (N, Q)
    y: torch.Tensor of shape (N, 1)
    """
    # Compute the quantile loss for each sample and each quantile
    loss = torch.max(tau * (y - pred_quantiles),
                     (1 - tau) * (pred_quantiles - y))
    # Compute the integral of the quantile loss
    return 2 * loss.sum(dim=1) / tau.shape[0]


if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("-f", "--fair", action="store_true",
                        help="Include the FAIR method on random samples.")
    # Test 1 - Two quantiles, observation is in the middle
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
    # Given a distribution N(0, 1), we'll create N evenly spaced observations y.
    # For each observation, we'll compute the CRPS using the 5 methods and
    # plot the error between the methods and the closed-form CRPS.
    # We'll do all of that for different numbers of quantiles Q.
    Q_values = [10, 32, 100, 1000]
    N = 200  # Number of observations to evaluate
    y = torch.linspace(-4, 4, N).view(N, 1)  # Observations
    mu = torch.zeros(N, 1)
    sigma = torch.ones(N, 1)
    dist = Normal(mu, sigma)  # N(0, 1) for each observation
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for Q, ax in zip(Q_values, axes.flatten()):
        tau = torch.linspace(1 / Q, 1, Q) - 0.5 / Q # Optimal quantiles
        # Method 1 (if requested)
        if parser.parse_args().fair:
            samples = dist.sample((Q,)).view(N, Q)
            crps_fair = fair_crps(samples, y)
        # Method 2
        crps_closed_form = normal_crps(mu, sigma, y).view((N,))
        # Method 3
        quantiles = dist.icdf(tau)
        crps_quantiles = QuantilesCRPS(tau)(quantiles.view(N, 1, -1), y.view(N, 1, 1),
                                            reduce_mean=False)
        # Method 4
        crps_ql_integral = quantiles_integral_crps(tau, quantiles, y)
        # Method 5
        crps_nrg = nrg_crps(quantiles, y)
        # Plot the results
        df = pd.DataFrame({
            "Closed-form": crps_closed_form.numpy(),
            "Quantiles, linear interp, INT": crps_quantiles.numpy(),
            "Integral of pinball loss": crps_ql_integral.numpy(),
            "NRG": crps_nrg.numpy()
        })
        if parser.parse_args().fair:
            df["Fair with i.i.d. samples"] = crps_fair.numpy()
        # Compute the error between the methods and the closed-form CRPS
        df = df - df["Closed-form"].values.reshape(N, 1)
        df["y"] = y.numpy().reshape(N)
        df = df.melt(id_vars="y", var_name="Method", value_name="CRPS Error")
        sns.lineplot(data=df, x="y", y="CRPS Error", hue="Method", ax=ax)
        # Plot the first and last quantiles as vertical lines
        ax.axvline(x=quantiles[0, 0].item(), color="black", linestyle="--")
        ax.axvline(x=quantiles[0, -1].item(), color="black", linestyle="--")
        ax.set_title(f"Q = {Q}")
    # Disable the legend for all subplots but the first one
    for ax in axes.flatten()[1:]:
        ax.get_legend().remove()
    fig.suptitle("Error between the CRPS estimation methods and the closed-form CRPS")
    plt.savefig('figures/examples/crps_estimation.png')
