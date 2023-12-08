"""
Implements the DeterministicDistribution class, which is useful to integrate a deterministic
model into the probabilistic pipeline.
"""
import torch


def flatten_MSE(y_pred, y_true):
    """
    Flattens the tensor and computes the MSE.
    """
    return torch.mean((y_pred.flatten() - y_true.flatten()) ** 2)


def flatten_MAE(y_pred, y_true):
    """
    Flattens the tensor and computes the MAE.
    """
    return torch.mean(torch.abs(y_pred.flatten() - y_true.flatten()))


def flatten_RMSE(y_pred, y_true):
    """
    Flattens the tensor and computes the RMSE.
    """
    return torch.sqrt(torch.mean((y_pred.flatten() - y_true.flatten()) ** 2))


class DeterministicDistribution:
    """
    Object that defines a deterministic distribution, i.e. the CDF is the step function
    that assigns probability 1 after the predicted value, and 0 before it.
    """
    def __init__(self):
        # The distribution P(y|x) is deterministic, so it is characterized by a single
        # parameter, which is the predicted value.
        self.n_parameters = 1

        # The output of the model will have shape (N, 1), where N is the batch size, and
        # 1 is because the "distribution" has only one parameter.
        # Thus we must flatten the output of the model before computing the loss.
        self.loss_function = flatten_MSE
        
        # Define the metrics
        self.metrics = {
                'RMSE': flatten_RMSE,
                'MAE': flatten_MAE,
                'CRPS': flatten_MAE  # The CRPS is the MAE for a deterministic distribution
                }
    
    def hyperparameters(self):
        """
        Returns the hyperparameters of the distribution.
        """
        return {}
    
    def cdf(self, y_pred, y):
        """
        Given a deterministic distribution that always outputs y, the CDF is defined as
        1 for y_pred >= y, and 0 otherwise.
        """
        return torch.heaviside(y - y_pred, torch.tensor([1.0]))

    def inverse_cdf(self, y, u):
        """
        Given a deterministic distribution that always outputs y, the inverse CDF is
        defined as y for any u in [0, 1].
        """
        return y

