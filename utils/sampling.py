"""
Defines objects for the sampling strategy.
"""
import torch
import scipy.stats as ss
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler


def inverse_intensity_sampler(intensities, plot_weights=None):
    """
    Returns a torch Sampler object that samples indices with probability
    inversely proportional to the intensity of the samples.
    The intensity of a sample with multiple time steps is defined as the
    maximum intensity over all time steps.
    As the intensities are not exactly continuous in practice, a probability

    
    Parameters
    ----------
    intensities : torch.Tensor of shape (N, T)
        The intensity of each sample in the dataset. The first dimension
        corresponds to the number of samples, and the second dimension
        corresponds to the number of time steps.
    plot_weights : str or None
        If not None, the path to save a plot of the weights of the samples.

    Returns
    -------
    torch.utils.data.Sampler
        A sampler object that can be used to sample indices from the dataset.
    """
    # Compute the maximum intensity over all time steps 
    max_intensities, _ = torch.max(intensities, dim=1)
    # Convert the intensities to a numpy array for compatibility with scipy
    max_intensities = max_intensities.numpy()
    # Fit a gamma distribution to the intensities
    params = ss.gamma.fit(max_intensities)
    # Compute the pdf at the intensity of each sample
    probs = ss.gamma.pdf(max_intensities, *params)
    # Deduce the weights as the inverse of the pdf
    weights = 1 / probs
    # There's no need to normalize the weights, as they will be normalized by
    # the torch sampler anyway
    # Plot the weights if requested
    if plot_weights is not None:
        # On the same plot, show:
        # * The histogram of the maximum intensities
        # * The fitted gamma distribution
        # * The weights, which are the inverse of the fitted gamma distribution,
        #   with a different y axis
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        # Plot an histogram of the maximum intensities
        ax1.hist(max_intensities, density=True, alpha=0.5, label="Empirical max intensities distribution")
        # Plot the fitted gamma distribution
        x = torch.linspace(1, max(max_intensities), 100)
        y = ss.gamma.pdf(x, *params)
        ax1.plot(x, y, label="Fitted gamma distribution")
        # Plot the weights
        ax2.plot(x, 1 / y, 'r.', label="Weights")
        # Disable the grid for the weights
        ax2.grid(False)
        ax1.set_xlabel("Max intensity")
        ax1.set_ylabel("Density")
        ax2.set_ylabel("Weights")
        fig.legend()
        plt.savefig(plot_weights)

    # Create the sampler
    return WeightedRandomSampler(weights, intensities.shape[0], replacement=True)
