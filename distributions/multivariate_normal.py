"""
 the multivariate normal distribution, which can be used to model
P(Y_{1:T} | X_{-P+1:0}).
"""
import torch
from distributions.normal import normal_crps


class MultivariateNormal:
    """
    Class that implements the loss function and metrics for a model that predicts the parameters
    of a multivariate normal distribution.

    Parameters
    ----------
    dim: int
        Dimension of the multivariate normal distribution (for forecasting a time series,
        number of time steps to predict).
    """
    def __init__(self, dim):
        self.dim = dim
        self.is_multivariate = True
        # The parameters of the multivariate normal distribution are the mean and the
        # covariance matrix. The covariance matrix is symmetric, so we only need to
        # predict the upper triangular part.
        self.n_parameters = dim + dim * (dim + 1) // 2

        # Metrics
        self.metrics = {
                'nll': self.loss_function,
                'distance_to_mean_rmse': self.distance_to_mean_rmse,
                'CRPS': self.marginal_crps
        }

    def loss_function(self, predicted_params, y):
        """
        Computes the negative log likelihood of the multivariate normal distribution:
        L(P, y) = -log(P(y))

        Parameters
        ----------
        predicted_params:  Pair (mean, L) where:
            - mean: torch.Tensor of shape (N, dim)
                Mean vector of the distribution.
            - L: torch.Tensor of shape (N, dim, dim)
                Cholesky factor of the covariance matrix.
        y: torch.Tensor of shape (N, dim)
            Observed values.

        Returns
        -------
        loss: float 
            Negative log likelihood of the multivariate normal distribution.
        """
        mean, L = predicted_params
        # Build the distributions using torch.distributions
        # Remark: giving the Cholesky factor to torch instead of the covariance matrix
        # is more efficient, as torch computes the Cholesky factor internally anyway.
        # See https://pytorch.org/docs/stable/distributions.html#multivariatenormal .
        dist = torch.distributions.MultivariateNormal(mean, scale_tril=L)
        # Compute the negative log likelihood of the distribution.
        loss = -dist.log_prob(y)
        return loss.mean()

    def activation(self, predicted_params):
        """
        Reconstructs the mean vector and the Cholesky factor of the covariance matrix
        from the predicted parameters, and applies the Softplus activation to the diagonal
        of the Cholesky factor.

        Parameters
        ----------
        predicted_params: torch.Tensor of shape (N, dim + dim * (dim + 1) // 2)
            Predicted parameters of the multivariate normal distribution, where the order
            of the parameters is:
            - 0 to dim - 1: mean of the distribution.
            - dim to 2*dim - 1: diagonal of the cholesky factor of the covariance matrix.
                (Must be positive.)
            - 2*dim to 2*dim + dim*(dim - 1) // 2 - 1: lower triangular part of the cholesky
                factor.

        Returns
        -------
        mean: torch.Tensor of shape (N, dim)
            Mean vector of the distribution.
        L: torch.Tensor of shape (N, dim, dim)
            Cholesky factor of the covariance matrix.
        """
        # Split the predicted parameters into the mean and the cholesky factor of the
        # covariance matrix.
        mean = predicted_params[:, :self.dim]
        # Retrieve the diagonal and lower triangular part of the cholesky factor.
        diag = predicted_params[:, self.dim:2*self.dim]
        # Apply the Softplus activation to the diagonal.
        diag = torch.nn.functional.softplus(diag)
        # Retrieve the lower triangular part of the cholesky factor.
        lower_triangular = predicted_params[:, 2*self.dim:]
        # Construct the cholesky matrix L from the diagonal and lower triangular part.
        # The covariance matrix is L L^T.
        L = torch.diag_embed(diag)
        indices = torch.tril_indices(self.dim, self.dim, offset=-1)
        L[:, indices[0], indices[1]] = lower_triangular
        return mean, L

    def denormalize(self, predicted_params, task, dataset):
        """
        De-normalizes the predicted parameters of the distribution.

        Parameters
        ----------
        predicted_params : Pair (mean, L) where:
            - mean: torch.Tensor of shape (N, dim)
                Mean vector of the distribution.
            - L: torch.Tensor of shape (N, dim, dim)
                Cholesky factor of the covariance matrix.
        task : str
            The name of the task.
        dataset : dataset, as an object that implements the
            get_normalization_constants method.

        Returns
        -------
        new_mean: torch.Tensor of shape (N, dim)
            De-normalized mean vector of the distribution.
        new_L: torch.Tensor of shape (N, dim, dim)
            De-normalized Cholesky factor of the covariance matrix.
        """
        pred_mean, pred_L = predicted_params
        # Get the normalization constants from the tasks dictionary
        means, stds = dataset.get_normalization_constants(task)
        # Move the normalization constants to the same device as the predicted parameters.
        means, stds = means.to(pred_mean.device), stds.to(pred_mean.device)
        # The denormalization is done as such:
        # Let S = diag(stds) and m = means.
        # Then new_mean = S pred_mean + m
        # and new_L = S pred_L
        new_mean = stds * pred_mean + means
        new_L = stds.unsqueeze(-1) * pred_L
        return new_mean, new_L

    def distance_to_mean_rmse(self, predicted_params, y):
        """
        Computes the RMSE between the mean of the distribution and the observed values.

        Parameters
        ----------
        predicted_params: Pair (mean, L) where:
            - mean: torch.Tensor of shape (N, dim)
                Mean vector of the distribution.
            - L: torch.Tensor of shape (N, dim, dim)
                Cholesky factor of the covariance matrix.
        y: torch.Tensor of shape (N, dim)
            Observed values.

        Returns
        -------
        rmse: float 
            Average RMSE between the mean of the distribution and the observed values.
        """
        mean, _ = predicted_params
        return torch.sqrt(torch.mean((mean - y)**2))

    def marginal_crps(self, predicted_params, y):
        """
        Computes the average CRPS over each marginal distribution.

        Parameters
        ----------
        predicted_params: Pair (mean, L) where:
            - mean: torch.Tensor of shape (N, dim)
                Mean vector of the distribution.
            - L: torch.Tensor of shape (N, dim, dim)
                Cholesky factor of the covariance matrix.
        y: torch.Tensor of shape (N, dim)
            Observed values.

        Returns
        -------
        crps: float
            Average CRPS over each marginal distribution.
        """
        mean, L = predicted_params
        # Retrieve the standard deviation of each marginal distribution
        std = torch.diagonal(L, dim1=-2, dim2=-1)
        # Compute the CRPS for each marginal distribution
        crps = normal_crps(mean, std, y)
        return crps.mean()
