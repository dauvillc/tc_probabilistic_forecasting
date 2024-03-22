"""
Implements the ResidualPrediction class, which allows to manage
predictions that are split between location and residuals.
"""

import os
import torch
from pathlib import Path
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

    def add(self, task_name, location, residual_params, distrib=DeterministicDistribution()):
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
        distrib: PredictionDistribution, optional
            Distribution used to predict the residuals. Defaults to a deterministic
            distribution.

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
            new_res = self.distribs[task_name].denormalize(
                self.residuals[task_name], task_name, dataset, is_residuals=True
            )
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

    def keys(self):
        return self.locations.keys()

    def save(self, save_dir):
        """
        Saves the predictions to disk.

        Parameters
        ----------
        save_dir : str
            The directory where to save the predictions.
        """
        save_dir = Path(save_dir)
        locations_path = save_dir / "locations"
        residuals_path = save_dir / "residuals"
        final_path = save_dir / "final"
        os.makedirs(locations_path, exist_ok=True)
        os.makedirs(residuals_path, exist_ok=True)
        os.makedirs(final_path, exist_ok=True)
        for task_name in self.locations.keys():
            torch.save(self.locations[task_name], locations_path / f"{task_name}.pt")
            torch.save(self.residuals[task_name], residuals_path / f"{task_name}.pt")
            torch.save(self.final[task_name], final_path / f"{task_name}.pt")

    @staticmethod
    def cat(predictions):
        """
        Concatenates multiple ResidualPrediction objects.

        Parameters
        ----------
        predictions : list of ResidualPrediction
            The list of predictions to concatenate.

        Returns
        -------
        ResidualPrediction
            The concatenated predictions.
        """
        cat_preds = ResidualPrediction()
        for task_name in predictions[0].keys():
            locations = torch.cat([pred.locations[task_name] for pred in predictions], dim=0)
            residuals = torch.cat([pred.residuals[task_name] for pred in predictions], dim=0)
            distrib = predictions[0].distribs[task_name]
            cat_preds.add(task_name, locations, residuals, distrib)
        return cat_preds
