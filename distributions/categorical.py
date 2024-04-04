"""
Implementation of the CategoricalDistribution class,
which allows to perform classification tasks.
"""

import torch
from distributions.prediction_distribution import PredictionDistribution
from utils.utils import average_score


class CategoricalDistribution(PredictionDistribution):
    """
    Models classification tasks, using the cross-entropy loss.
    
    Parameters
    ----------
    num_classes : int
        Number of classes in the classification task.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.n_parameters = num_classes
        self.is_multivariate = False
        self.metrics = {}

    def activation(self, predicted_params):
        """
        Applies the softmax function to the predicted parameters to
        convert them into probabilities.

        Parameters
        ----------
        predicted_params : torch.Tensor
            The predicted parameters of the distribution, as a tensor of shape
            (N, T, num_classes).
        """
        return torch.nn.functional.softmax(predicted_params, dim=-1)

    def loss_function(self, predicted_params, target, reduce_mean="all"):
        """
        Computes the Cross-entropy loss between the predicted parameters and the target.

        Parameters
        ----------
        predicted_params : torch.Tensor
            The predicted parameters of the distribution, as a tensor of shape
            (N, T, num_classes).
        target : torch.Tensor
            The target values, as a tensor of shape (N, T).
        reduce_mean : str
            Over which dimension(s) to average the loss. Can be "all", "samples", "time" or "none".

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        # Flatten the batch and time dimensions
        N, T = target.shape
        predicted_params = predicted_params.view(-1, self.num_classes)
        target = target.view(-1)
        # Convert the target to integer type
        target = target.long()
        # Convert the probabilities to log-probas
        log_probs = predicted_params.log()
        # Apply the NLL loss to obtain the cross-entropy
        loss = torch.nn.functional.nll_loss(log_probs, target, reduction="none")
        # Reshape the loss to the original shape
        loss = loss.view(N, T)
        return average_score(loss, reduce_mean)
    
    def accuracy(self, predicted_params, target, reduce_mean="all"):
        """
        Computes the accuracy of a set of predicted probabilities.

        Parameters
        ----------
        predicted_params : torch.Tensor
            The predicted parameters of the distribution, as a tensor of shape
            (N, T, num_classes).
        target : torch.Tensor
            The target values, as a tensor of shape (N, T).
        reduce_mean : str
            Over which dimension(s) to average the accuracy. Can be "all", "samples", "time" or "none".
        """
        # Compute the predicted classes
        predicted_classes = torch.argmax(predicted_params, dim=-1)
        # Compute the accuracy
        accuracy = (predicted_classes == target).float()
        return average_score(accuracy, reduce_mean)

    def denormalize(self, predicted_params, task, dataset, is_residuals=False):
        return predicted_params

    def translate(self, predicted_params, x):
        return predicted_params

    def pdf(self, predicted_params, x):
        raise NotImplementedError("The Categorical distribution does not have a PDF.")

    def cdf(self, predicted_params, x):
        raise NotImplementedError("The Categorical distribution does not have a CDF.")
