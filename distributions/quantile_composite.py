"""
Defines the QuantileCompositeDistribution class.
"""
import numpy as np
import torch
from metrics.loss_functions import MultipleQuantileLoss
from metrics.quantiles import Quantiles_eCDF, Quantiles_inverse_eCDF, QuantilesCRPS


class QuantileCompositeDistribution:
    """
    Object that can define a distribution from a set of quantiles.

    Parameters
    ----------
    min_value : float
        The minimum value of the distribution.
    max_value : float
        The maximum value of the distribution.
    """
    def __init__(self, min_value, max_value):
        self.quantiles = np.linspace(0.01, 0.99, 99)
        self.n_parameters = len(self.quantiles)
        self.min_value = min_value
        self.max_value = max_value
        
        # Define the loss function
        self.loss_function = MultipleQuantileLoss(self.quantiles)

        # Define the metrics
        self.metrics = {}
        # First, the Composite Quantile Loss for increasingly higher quantiles
        min_quantiles = [0.5, 0.75, 0.9]
        for q in min_quantiles:
            self.metrics[f"CQL_{q}"] = MultipleQuantileLoss(self.quantiles, min_quantile=q)
        # Then, the MAE (corresponding to the 0.5 quantile)
        self.metrics["MAE"] = MultipleQuantileLoss([0.5])
        # Then, the CRPS
        self.metrics["CRPS"] = QuantilesCRPS(self.quantiles, min_value, max_value)

        # Define the CDF and inverse CDF
        self.cdf = Quantiles_eCDF(self.quantiles, min_value, max_value)
        self.inverse_cdf = Quantiles_inverse_eCDF(self.quantiles, min_value, max_value) 

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

    def hyperparameters(self):
        """
        Returns the hyperparameters of the distribution. Here, it is
        the minimum and maximum values of the distribution, as well as
        the quantiles defining the distribution.

        Returns
        -------
        hyperparameters : dict
            The hyperparameters of the distribution.
        """
        return {"min_value": self.min_value,
                "max_value": self.max_value,
                "quantiles": self.quantiles}

