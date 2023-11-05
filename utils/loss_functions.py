"""
Implementation of loss functions.
"""
import matplotlib.pyplot as plt
import torch


class WeightedMSELoss(torch.nn.Module):
    """
    Implements a weighted mean squared error loss function.
    The intensities are divided into bins. The weight of each bin is 1 / probability
    of the bin.
    
    Parameters
    ----------
    all_intensities : array-like
        Intensities of all samples in the dataset.
    """
    def __init__(self, all_intensities):
        super().__init__()
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
        # Use the square root of the weights, as they will be squared in the loss
        self.weights = torch.sqrt(self.weights)
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
        loss = torch.mean(weights * (y_pred - y_true) ** 2)
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
