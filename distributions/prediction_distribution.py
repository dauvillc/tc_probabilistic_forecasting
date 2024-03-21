"""
Implements the abstract class PredictionDistribution, which represents a
parametric distribution that can be used to probabilistically predict the
target values.
"""

from abc import ABC, abstractmethod


class PredictionDistribution(ABC):
    """
    Represents a parametric distribution that can be used to probabilistically
    predict the target values of a target variable V at different time steps.
    """

    def __init__(self):
        # Abstract attributes which will have to be implemented by the subclasses
        # Number of parameters that define the distribution (e.g. 2 for a normal distribution)
        n_parameters: int
        # Whether the distribution is multivariate or not. A multivariate distribution
        # is the joint over all time steps, while non-multivariate distributions characterize
        # the marginal distribution at each time step.
        is_multivariate: bool
        # Map metric_name -> function F(y_pred, y_true, reduce_mean="all") -> score
        metrics: dict

    def activation(self, predicted_params):
        """
        Method that will be applied to the output of the network to convert
        into distribution parameters (e.g. softplus to ensure positivity).
        """
        return predicted_params

    @abstractmethod
    def loss_function(self, predicted_params, y, reduce_mean="all"):
        """
        Computes the loss function for the distribution.
        """
        pass

    @abstractmethod
    def denormalize(self, predicted_params, task, dataset, is_residuals=False):
        """
        Denormalizes the predicted parameters. The process necessarily depends
        on the distribution.
        The task and dataset arguments are used to access the normalization
        constants.
        """
        pass

    @abstractmethod
    def translate(self, predicted_params, x):
        """
        Translates the distribution by a given amount.
        """
        pass

    @abstractmethod
    def pdf(self, predicted_params, x):
        """
        Computes the probability density function of the distribution at x.
        """
        pass

    @abstractmethod
    def cdf(self, predicted_params, x):
        """
        Computes the cumulative distribution function of the distribution at x.
        """
        pass
