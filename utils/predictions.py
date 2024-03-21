"""
Implements the ResidualPrediction class, which allows to manage
predictions that are split between location and residuals.
"""
from distributions.deterministic import DeterministicDistribution


class ResidualPrediction:
    """
    Manages predictions that are split as:
    - The predicted location of the first time step Ŷ_0;
    - The residuals between the predicted location and the next time steps, i.e.
      (Ŷ_1 - Y_0), (Ŷ_2 - Y_0), ..., (Ŷ_T - Y_0).
    The locations are predicted deterministically, while the residuals are predicted
    via a parametric distribution.
    The class manages predictions for multiple tasks, each with its own predictions.

    Parameters
    ----------
    """
    def __init__(self):
        # Dictionaries to store the predictions
        # Keys are task names
        # locations: tensors of shape (batch_size, 1)
        self.locations = {}
        # residuals: tensors of shape (batch_size, T, n_parameters)
        self.residuals = {}
        # distribs: PredictionDistribution objects used to predict the residuals
        self.distribs = {}
        # final: sums of locations and residuals
        self.final = {}
        # Determinstic distribution object which will be used to treat the locations
        self.loc_distrib = DeterministicDistribution()

    def add(self, task_name, location, residual_params, distrib):
        """
        Adds the predictions for a given task.
        Automatically calls the activation function of the residual distribution.

        Parameters
        ----------
        task_name : str
            The name of the task.
        location : torch.Tensor
            The predicted location of the first time step Ŷ_0, as a tensor of shape
            of shape (batch_size, 1).
        residual_params: torch.Tensor
            The parameters of the distribution that predicts the residuals, as a tensor
            of shape (batch_size, T, n_parameters).
        distrib: PredictionDistribution
            Distribution used to predict the residuals.

        Raises
        ------
        ValueError
            If the task_name is already in the predictions.
        """
        if task_name in self.locations:
            raise ValueError(f"The task {task_name} is already in the predictions.")
        self.locations[task_name] = location
        # Store the distribution used to predict the residuals
        self.distribs[task_name] = distrib
        # Apply the activation function to the residuals
        self.residuals[task_name] = distrib.activation(residual_params)
        # Compute the distribution of Ŷ_t based on Ŷ_0 and the residuals
        self.final[task_name] = distrib.translate(residual_params, location)

    def denormalize(self, dataset):
        """
        Returns a new object with the predictions denormalized.

        Parameters
        ----------
        dataset: SuccessiveStepsDataset
            Pointer to the dataset used to retrieve the normalization parameters.

        Returns
        -------
        ResidualPrediction
            A new object with the predictions denormalized.
        """
        denorm_preds = ResidualPrediction()
        for task_name in self.locations.keys():
            # The denormalization is specific to each distribution
            new_loc = self.loc_distrib.denormalize(self.locations[task_name], task_name, dataset)
            new_res = self.distribs[task_name].denormalize(self.residuals[task_name], task_name, dataset)
            denorm_preds.add(task_name, new_loc, new_res, self.distribs[task_name])
        return denorm_preds

    def final_predictions(self, task_name):
        """
        Returns the parameters of the final distribution for a given task, i.e.
        the distribution of Ŷ_0 + residuals.

        Parameters
        ----------
        task_name : str
            The name of the task.

        Returns
        -------
        torch.Tensor
            The parameters of the distribution of Ŷ_t, as a tensor of shape
            (batch_size, T, n_parameters).
        """
        return self.final[task_name] 

