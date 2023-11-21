"""
Implementation of loss functions.
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """
    Implements the standard quantile loss function.
    
    Parameters
    ----------
    quantile : float
        The quantile to be estimated.
    reduction: str, optional.
        Can be either 'none', 'mean' or 'sum'.
        Defaults to 'mean'.
    """
    def __init__(self, quantile, reduction="none"):
        super().__init__()
        self.quantile = quantile
        if reduction == "none":
            self.reduction = lambda x: x
        elif reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        else:
            raise ValueError(f"The reduction argument '{reduction}' was not understood.")

    def __call__(self, y_pred, y_true):
        result = torch.max((1 - self.quantile) * (y_true - y_pred),
                           self.quantile * (y_pred - y_true))
        return self.reduction(result)


class MultipleQuantileLoss(nn.Module):
    """
    Implements the multiple quantile loss function.
    The prediction should have the shape (N, T, Q) where N is the batch size,
    T is the number of time steps, and Q is the number of quantiles.

    Parameters
    ----------
    quantiles : array-like of floats
        The quantiles to be estimated.
    weights: array-like of floats, optional
        The weights associated with each quantile.
        By default, no weights are applied.
    reduction: str, optional
        Either "none" (default), "mean" or "sum".
    """
    def __init__(self, quantiles, weights=None, reduction="none"):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)
        # The term (y_true - y_pred) will have shape (N, T, Q).
        self.quantiles = self.quantiles.unsqueeze(0)
        self.reduction = reduction
        if weights is not None:
            weights = torch.tensor(weights).unsqueeze(1)
        self.weights = weights
        # Variable to remember whether this has been called before
        self.called = False

    def __call__(self, y_pred, y_true):
        diff = y_true.unsqueeze(2) - y_pred
        # First call: adjust the shape of the quantiles and 
        # transfer the tensors to the same device as the inputs
        if not self.called:
            self.called = True
            self.quantiles = self.quantiles.expand_as(diff[0])
            self.quantiles = self.quantiles.to(y_true.device)
            if self.weights is not None:
                self.weights = self.weights.to(y_true.device)
        loss = torch.max(self.quantiles * diff, (self.quantiles - 1) * diff)
        # Apply the weights
        if self.weights is not None:
            loss = loss * self.weights
        # Apply the reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


class WeightedLoss(torch.nn.Module):
    """
    Implements a weighted loss function, in which the weights depend on the intensity of
    the storms.
    The intensities are divided into bins. The weight of each bin is 1 / probability
    of the bin.
    
    Parameters
    ----------
    all_intensities : array-like
        Intensities of all samples in the dataset.
    base_loss: function of the form (y_pred, y_true) -> torch.Tensor
        Base loss function. The returned tensor should be of shpae (N,) where N is the batch size,
        i.e. no reduction is applied.
        Default is the mean squared error.
    weight_capping_intensity: float or None, optional
        If not None, the weights will remain constant for intensities
        greather or equal to this value.
    """
    def __init__(self, all_intensities, base_loss=None, weight_capping_intensity=None):
        super().__init__()
        if base_loss is None:
            base_loss = nn.MSELoss(reduction="none")
        self.base_loss = base_loss
        # Divide the intensities in 100 bins
        bins = 20
        # Convert the intensities to a tensor
        self.all_intensities = torch.tensor(all_intensities)
        # Compute the histogram of the intensities
        probs, bin_edges = torch.histogram(self.all_intensities,
                                           range=(0, 100),
                                           bins=bins, density=True)
        # Compute the weights of each bin as 1 / probability
        # when the probability is non-zero, and 0 otherwise.
        self.weights = torch.where(probs != 0, 1 / probs, torch.zeros_like(probs))
        # If the weight capping intensity is not None:
        # - Find the index of the bin corresponding to the weight capping intensity
        # - Fetch the weight w of this bin
        # - Cap all the weights to w
        if weight_capping_intensity is not None:
            weight_capping_bin_index = torch.bucketize(torch.tensor(weight_capping_intensity),
                                                       bin_edges) - 1
            max_weight = self.weights[weight_capping_bin_index]
            self.weights[self.weights > max_weight] = max_weight
        # Normalize the weights
        self.weights = self.weights / torch.sum(self.weights)
        # Store the bin edges for later use
        self.bin_edges = bin_edges

    def __call__(self, y_pred, y_true):
        """
        Compute the weighted mean squared error loss.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted values.
        y_true : torch.Tensor
            The true values.
        
        Returns
        -------
        torch.Tensor
            The loss.
        """
        # Bring back the tensors to the CPU if they are on the GPU
        # The reason we do this is that the bucketize function is not implemented
        # on the GPU
        y_true_cpu = y_true.cpu()
        # Compute the weights of the samples
        weights = self.weights[torch.bucketize(y_true_cpu, self.bin_edges) - 1]
        # Load the weights on the same device as the tensors
        weights = weights.to(y_true.device)
        # Compute the loss
        loss = torch.mean(weights * self.base_loss(y_pred, y_true))
        return loss
    
    def plot(self):
        """
        Plot the histogram of the intensities and the weights.
        """
        # Plot the distribution of the intensities and the weights
        # on the same plot with two different y-axes
        fig, ax1 = plt.subplots()
        ax1.hist(self.all_intensities, bins=100, density=True)
        ax1.set_ylabel('Probability density')
        ax2 = ax1.twinx()
        # only plot the weights for non-zero values
        ax2.plot(self.bin_edges[:-1][self.weights != 0], self.weights[self.weights != 0], '-o', color='orange')
        ax2.set_ylabel('Weight')
        plt.savefig('figures/examples/weighed_mse_loss.png')
