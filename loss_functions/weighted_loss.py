"""
Implements a weighted loss function.
"""

import matplotlib.pyplot as plt
import torch
import scipy.stats as ss
import numpy as np
from torch.distributions import LogNormal


class WeightedLoss:
    """
    Allows to apply sample weights to a loss function.
    The weights are inversely proportional to the intensity of the samples.
    Since the intensity is not exactly continuous in practice, a Gamma
    distribution is fitted to the intensities, and the inverse of the pdf
    is used as the weights.

    Parameters
    ----------
    train_dataset: SuccessiveStepsDataset
        The training dataset, implementing the get_sample_intensities and
        get_normalization_constants methods.
    test_goodness_of_fit : bool, optional
        If True, perform a Kolmogorov-Smirnov test to check if the fitted
        distribution is a good fit to the intensities.
        Default is False._smalll
    plot_weights : str or None, optional
        If not None, the path to save a plot of the weights of the samples.
    """

    def __init__(self, train_dataset, test_goodness_of_fit=False, plot_weights=None):
        # Save a pointer to the training dataset
        self.train_dataset = train_dataset
        print("Setting up weighted loss")
        # Get the intensities of the training samples
        intensities = train_dataset.get_sample_intensities()
        # Compute the maximum intensity over all time steps
        max_intensities, _ = torch.max(intensities, dim=1)
        # Convert the intensities to a numpy array for compatibility with scipy
        max_intensities = max_intensities.numpy()
        # The intensities are not exactly continuous, so we add a small amount
        # of noise to make the empirical distribution easier to fit.
        max_intensities += np.random.normal(0, 5, max_intensities.shape)
        # Intensities below 12kts are not impactful but are very rare, so they would
        # get a very high weight. To avoid this, we clip the intensities to 10kts minimum
        # when computing the weights.
        max_intensities = np.clip(max_intensities, 12, None)
        # Fit a weibull distribution to the intensities. When the loss is called,
        # the pdf will be used to compute the weights.
        shape, loc, scale = ss.lognorm.fit(max_intensities, floc=0)
        self.scale, self.shape = np.log(scale), shape
        # Create a torch distribution from the fitted parameters.
        distribution = LogNormal(self.scale, shape)
        # We'll now compute the sum of the weights, so that we can normalize them
        # to sum to 1.
        y = distribution.log_prob(torch.tensor(max_intensities)).exp()
        self.weights_integral = (1 / y).sum().item()

        # Test the goodness of fit if requested
        if test_goodness_of_fit:
            # Perform a Kolmogorov-Smirnov test to check if the fitted
            # distribution is a good fit to the intensities
            _, p_value = ss.kstest(max_intensities, "lognorm", args=(shape, 0, scale))
            print(f"Kolmogorov-Smirnov test p-value: {p_value}")

        # Plot the weights if requested
        if plot_weights is not None:
            # On the same plot, show:
            # * The histogram of the maximum intensities
            # * The fitted distribution
            # * The weights, which are the inverse of the fitted distribution,
            #   with a different y axis
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            # Plot an histogram of the maximum intensities
            ax1.hist(
                max_intensities,
                density=True,
                bins=100,
                alpha=0.5,
                label="Empirical max intensities distribution",
            )
            # Plot the fitted distribution
            x = torch.linspace(min(max_intensities), max(max_intensities), 100)
            y = distribution.log_prob(x).exp().numpy()
            ax1.plot(x, y, label="Fitted distribution")
            # Plot the weights
            ax2.plot(x, (1 / y) / self.weights_integral, "r.", label="Weights")
            # Disable the grid for the weights
            ax2.grid(False)
            ax1.set_xlabel("Max intensity")
            ax1.set_ylabel("Density")
            ax2.set_ylabel("Weights")
            fig.legend()
            plt.savefig(plot_weights)
        print("Weighted loss set up")

    def __call__(self, loss_values, intensities, reduce_mean=True):
        """
        Applies the weights to a tensor of loss values.

        Parameters
        ----------
        loss_values : torch.Tensor of shape (N,)
            The loss values to be weighted.
        intensities : torch.Tensor of shape (N,)
            The intensity of each sample in the batch.
        reduce_mean : bool
            If True, the loss is averaged over the batch. If False, returns
            a tensor of shape (n,), where n is the number of samples in the batch.

        Returns
        -------
        torch.Tensor
            The loss value.
        """
        device = loss_values.device
        # Now take the max intensity over all time steps
        intensities, _ = torch.max(intensities, dim=1)
        # Clip the intensities (see __init__ for explanation)
        intensities = torch.clamp(intensities, 12, None)
        # Create a torch distribution from the fitted parameters.
        distribution = LogNormal(
            torch.tensor(self.scale, device=device),
            torch.tensor(self.shape, device=device),
        )
        probas = distribution.log_prob(intensities).exp()
        # Compute the weights using the fitted distribution
        weights = (1 / probas) / self.weights_integral
        # Apply the weights to the loss values
        weighted_loss = loss_values * weights
        if reduce_mean:
            return torch.mean(weighted_loss)
        else:
            return weighted_loss
