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

    Parameters
    ----------
    tasks: dict
        Pointer to the tasks dictionary, which contains the normalization constants.
    """
    def __init__(self, tasks):
        self.tasks = tasks
        # The distribution P(y|x) is deterministic, so it is characterized by a single
        # parameter, which is the predicted value.
        self.n_parameters = 1

        # The output of the model will have shape (N, T, V) where N is the batch size,
        # T is the number of time steps and V is the number of output variables.
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
    
    def denormalize(self, predicted_params, task):
        """
        Denormalizes the predicted values.

        Parameters
        ----------
        predicted_params : torch.Tensor of shape (N, T, V)
            The predicted values for each sample and time step.
        task : str

        Returns
        -------
        torch.Tensor of shape (N, T, V)
            The denormalized predicted values.
        """
        # Retrieve the normalization constants, of shape (T * V)
        means = torch.tensor(self.tasks[task]['means'].values, dtype=torch.float32)
        stds = torch.tensor(self.tasks[task]['stds'].values, dtype=torch.float32)
        # Reshape the means and stds to be broadcastable and move them to the same device
        # as the predictions
        means = means.view(predicted_params.shape[1:]).to(predicted_params.device)
        stds = stds.view(predicted_params.shape[1:]).to(predicted_params.device)
        # De-normalize the predictions
        return predicted_params * stds + means
