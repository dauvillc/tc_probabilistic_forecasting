"""
Implements the TasksValues class, which allows to manage
predictions that are split between location and residuals.
"""

import os
import torch
from pathlib import Path
from distributions.deterministic import DeterministicDistribution


class TasksValues:
    """
    Manages predictions or targets of several tasks. Optionally, the values
    can be divided into a location and residuals:
    - The location of the first time step Y_0;
    - The residuals between the location and the next time steps, i.e.
      (Y_1 - Y_0), (Y_2 - Y_0), ..., (Y_T - Y_0).
    The values represent distribution parameters (possibly a deterministic
    ditribution). For tasks that split the values, the locations are necessarily
    deterministic while the residuals can be modeled by any distribution.

    Parameters
    ----------
    """

    def __init__(self):
        # Dictionaries to store the values
        # Keys are task names
        # locations: tensors of shape (batch_size, 1)
        self.locations = {}
        # residuals: tensors of shape (batch_size, T, n_parameters)
        self.residuals = {}
        # distribs: PredictionDistribution objects used to predict the residuals
        self.distribs = {}
        # final: sums of locations and residuals or direct values
        self.final = {}
        # Determinstic distribution object which will be used to treat the locations
        self.det_distrib = DeterministicDistribution()

    def add(self, task_name, values, distrib=DeterministicDistribution()):
        """
        Stores values for a task.

        Parameters
        ----------
        task_name : str
            The name of the task.
        values : torch.Tensor
            The values to store, as a tensor of shape (batch_size, T, n_parameters).
        distrib: PredictionDistribution, optional
            Distribution used to model the values. Defaults to a deterministic
            distribution.
        """
        if task_name in self.locations:
            raise ValueError(f"The task {task_name} is already stored.")
        self.distribs[task_name] = distrib
        # Just store the values as final values
        self.final[task_name] = values

    def add_residual(self, task_name, locations, residual_params, distrib=DeterministicDistribution()):
        """
        Adds values for a task that are split between locations and residuals.
        The locations are just the predicted values for the first time step Y_0, while
        the residuals are parameters of a distribution that models the differences
        (Y_t - Y_0) for t > 0.

        Parameters
        ----------
        task_name : str
            The name of the task.
        locations : torch.Tensor
            The location of the first time step Y_0, as a tensor of shape
            of shape (batch_size, 1).
        residual_params: torch.Tensor
            The parameters of the distribution that predicts the residuals, as a tensor
            of shape (batch_size, T, n_parameters).
        distrib: PredictionDistribution, optional
            Distribution used to model the residuals. Defaults to a deterministic
            distribution.

        Raises
        ------
        ValueError
            If the task_name is already in the predictions.
        """
        if task_name in self.locations:
            raise ValueError(f"The task {task_name} is already stored.")
        self.locations[task_name] = locations
        # Store the distribution object
        self.distribs[task_name] = distrib
        # Compute the distribution of Y_t based on Y_0 and the residuals
        self.final[task_name] = distrib.translate(residual_params, locations)

    def denormalize(self, dataset):
        """
        Returns a new object with the values denormalized.

        Parameters
        ----------
        dataset: SuccessiveStepsDataset
            Pointer to the dataset used to retrieve the normalization parameters.

        Returns
        -------
        TasksValues
            A new object with the values denormalized.
        """
        denorm_values = TasksValues()
        # First, tasks that have only final values
        for task_name in self.final.keys():
            if task_name in self.locations:
                continue
            # The way the denormalization is done depends on the distribution
            new_values = self.distribs[task_name].denormalize(
                self.final[task_name], task_name, dataset, is_residuals=False
            )
            denorm_values.add(task_name, new_values, self.distribs[task_name])
        # Then, tasks that have locations and residuals
        for task_name in self.locations.keys():
            new_loc = self.det_distrib.denormalize(self.locations[task_name], task_name, dataset)
            new_res = self.distribs[task_name].denormalize(
                self.residuals[task_name], task_name, dataset, is_residuals=True
            )
            denorm_values.add_residual(task_name, new_loc, new_res, self.distribs[task_name])
        return denorm_values

    def final_predictions(self, task_name):
        """
        Returns the parameters of the final distribution for a given task, i.e.
        the distribution of Y_0 + residuals.

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
        return self.final.keys()

    def save(self, save_dir):
        """
        Saves the values to disk.

        Parameters
        ----------
        save_dir : str
            The directory where to save the values.
        """
        save_dir = Path(save_dir)
        locations_path = save_dir / "locations"
        residuals_path = save_dir / "residuals"
        final_path = save_dir / "final"
        os.makedirs(locations_path, exist_ok=True)
        os.makedirs(residuals_path, exist_ok=True)
        os.makedirs(final_path, exist_ok=True)
        for task_name in self.keys():
            if task_name in self.locations:
                torch.save(self.locations[task_name], locations_path / f"{task_name}.pt")
                torch.save(self.residuals[task_name], residuals_path / f"{task_name}.pt")
            torch.save(self.final[task_name], final_path / f"{task_name}.pt")

    @staticmethod
    def cat(tasks_values_objects):
        """
        Concatenates multiple ResidualPrediction objects.

        Parameters
        ----------
        tasks_values_objects : list of TasksValues
            The objects to concatenate.

        Returns
        -------
        TasksValues
            The concatenated object.
        """
        result = TasksValues()
        for task_name in tasks_values_objects[0].keys():
            # First, tasks that have only final values
            if task_name not in tasks_values_objects[0].locations:
                final_values = torch.cat([obj.final[task_name] for obj in tasks_values_objects], dim=0)
                result.add(task_name, final_values, tasks_values_objects[0].distribs[task_name])
            else:
                locations = torch.cat([obj.locations[task_name] for obj in tasks_values_objects], dim=0)
                residuals = torch.cat([obj.residuals[task_name] for obj in tasks_values_objects], dim=0)
                distrib = tasks_values_objects[0].distribs[task_name]
                result.add_residual(task_name, locations, residuals, distrib)
        return result

    @staticmethod
    def average(tasks_values_objects):
        """
        Averages the values stored in multiple TasksValues objects.

        Parameters
        ----------
        tasks_values_objects : list of TasksValues
            The objects to average.

        Returns
        -------
        TasksValues
            The averaged object.
        """
        result = TasksValues()
        for task_name in tasks_values_objects[0].keys():
            # First, tasks that have only final values
            if task_name not in tasks_values_objects[0].locations:
                # Stack the values
                final_values = torch.stack([obj.final[task_name] for obj in tasks_values_objects], dim=0)
                # Compute the average
                final_values = final_values.mean(dim=0)
                result.add(task_name, final_values, tasks_values_objects[0].distribs[task_name])
            else:
                # Do the same separately for locations and residuals
                locations = torch.stack([obj.locations[task_name] for obj in tasks_values_objects], dim=0)
                residuals = torch.stack([obj.residuals[task_name] for obj in tasks_values_objects], dim=0)
                locations = locations.mean(dim=0)
                residuals = residuals.mean(dim=0)
                distrib = tasks_values_objects[0].distribs[task_name]
                result.add_residual(task_name, locations, residuals, distrib)
        return result
