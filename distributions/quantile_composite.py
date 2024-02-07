"""
Defines the QuantileCompositeDistribution class.
"""
import torch
from loss_functions.quantiles import CompositeQuantileLoss, QuantilesCRPS


class QuantileCompositeDistribution:
    """
    Object that can define a distribution from a set of quantiles.

    Parameters
    ----------
    """
    def __init__(self):
        self.probas = torch.linspace(0.01, 0.99, 99)
        self.n_parameters = len(self.probas)
        self.is_multivariate = False
        
        # Define the loss function
        self.loss_function = CompositeQuantileLoss(self.probas)

        # Define the metrics
        self.metrics = {}
        # Then, the MAE (corresponding to the 0.5 quantile)
        self.metrics["MAE"] = CompositeQuantileLoss(torch.tensor([0.5]))
        # Then, the CRPS
        self.metrics["CRPS"] = QuantilesCRPS(self.probas)

    def activation(self, predicted_params):
        # Identity activation
        return predicted_params

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
                "probas": self.probas}

