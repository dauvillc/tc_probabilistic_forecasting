"""
Implements the SuccessiveStepsDataset class, which is a subclass of the
torch.utils.data.Dataset class. This class is used to yield successive
steps of a multiple time series, which can be either tabular data or images.
"""

import torch
import numpy as np
from torchvision.transforms import v2
from torchvision import tv_tensors


class SuccessiveStepsDataset(torch.utils.data.Dataset):
    """
    Given:
    - A Datacube X of shape (N, C, H, W), where N is the number of samples, C is the number of channels,
        H is the height and W is the width.
    - A tabular dataset S of storm trajectories of shape (N, K), where K is the number of variables.
    - A tasks dictionary, which maps task names to variables from S to predict.
    Returns elements of the form (X_i, S_i, Y_i), where:
    - X_i is a tensor of shape (P, C, H, W), where P is the number of past steps.
    - S_i is a tensor of shape (P, V), and contains the contextual information for X_i.
    - Y_i is a tensor of shape (T, V'), and contains the targets to predict.

    Parameters
    ----------
    subset: str
        "train", "val" or "test".
    trajectories: pandas.DataFrame
        The input time series. Must at least contain the columns "SID", "ISO_TIME", and
        the columns specified in input_columns and output_columns.
    input_columns: list of str
        The columns of the input time series, to be taken from trajectories.
    output_tabular_tasks: Mapping of str to Mapping
        The output tabular tasks. The keys are the names of the tasks, and the values
        are the parameters of the task, including at least:
            - 'output_variables': list of str
                The variables to predict.
            - 'predict_residuals': bool
                If True, the batches will yield Y_0 and the residuals Y_T - Y_0 instead of just
                Y_T.
    datacubes: Mapping of str to torch.Tensor
        The input datacubes. The keys are the names of the datacubes and the values are
        the datacubes themselves. Each datacube must be of shape (N, C, H, W), where N
        is the number of samples, one for each pair (SID, ISO_TIME).
        The datacubes should be already normalized.
    tabular_means: pd.Series
        The means of all variables in "trajectories".
    tabular_stds: pd.Series
        The standard deviations of all variables in "trajectories".
    datacube_means: Mapping of str to torch.Tensor
        DataArray of shape (C,) containing the means of each channel in the datacubes.
    datacube_stds: Mapping of str to torch.Tensor
        DataArray of shape (C,) containing the stds of each channel in the datacubes.
    cfg: dict
        The configuration dictionary.
    yield_ensemble: bool, optional
        If True, yields an ensemble of inputs in which the datacube is roated ten times
        by evenly spaced angles between 0 and 360 degrees. If False, defaults to whether
        random rotations are enabled in the configuration file.
    """

    def __init__(
        self,
        subset,
        trajectories,
        input_columns,
        output_tabular_tasks,
        datacubes,
        tabular_means,
        tabular_stds,
        datacube_means,
        datacube_stds,
        cfg,
        yield_ensemble=False,
    ):
        self.trajectories = trajectories
        self.input_columns = input_columns
        self.output_tabular_tasks = output_tabular_tasks
        self.datacubes = datacubes
        self.input_datacubes = list(datacubes.keys())
        self.tabular_means = tabular_means
        self.tabular_stds = tabular_stds
        self.datacube_means = datacube_means
        self.datacube_stds = datacube_stds
        self.past_steps = cfg["experiment"]["past_steps"]
        self.target_steps = cfg["experiment"]["target_steps"]
        self.patch_size = cfg["experiment"]["patch_size"]

        # Data augmentation: If requested, yield deterministic rotations of the datacubes
        # to create an ensemble of inputs.
        if yield_ensemble:
            self.random_rotations = np.linspace(0, 360, 10, endpoint=False)
        # Otherwise, apply a single random rotation to the datacubes if requested in the cfg
        # and only for the training set (no augmentation for val and test sets).
        else:
            self.random_rotations = "none"
            if cfg['training_settings']['data_augmentation'] == True:
                if subset == "train":
                    self.random_rotations = "random"

        if isinstance(self.random_rotations, np.ndarray):
            # If the data augmentation is not random, but a set of fixed rotations,
            # apply the rotations to the datacubes.
            self.transforms = [
                v2.Compose(
                    [
                        v2.RandomRotation(degrees=[angle, angle]),
                        v2.CenterCrop(self.patch_size),
                    ]
                )
                for angle in self.random_rotations
            ]
        elif self.random_rotations == "random":
            # Create the random transformations to apply to the datacubes.
            # The datacube is randomly rotated by angle between -180 and 179 degrees,
            # and is then center-cropped to 64x64 pixels.
            self.transforms = v2.Compose(
                [
                    v2.RandomRotation(degrees=(-180, 179)),
                    v2.CenterCrop(self.patch_size),
                ]
            )
        elif self.random_rotations == "none":
            # Otherwise, just crop the center of the datacube.
            self.transforms = v2.Compose([v2.CenterCrop(self.patch_size)])

    def denormalize_tabular_target(self, variables, residuals=False):
        """
        Denormalize a batch of target variables, using the constants stored in the dataset.

        Parameters
        ----------
        variables: Mapping of str to torch.Tensor
            The variables to denormalize. The keys are tasks names and the values are
            torch.Tensors.
        residuals: bool, optional
            If True, denormalizes the residuals between the target variables and the variable at t=0.
            Otherwise, denormalizes the target variables.

        Returns
        -------
        denormalized_variables: Mapping of str to torch.Tensor
            Structure identical to variables, but with the denormalized values.
        """
        denormalized_variables = {}
        for task in variables:
            # Retrieve the normalization constants for the task and convert them to tensors.
            means, stds = self.get_normalization_constants(task, residuals=residuals)
            # Load them to the same device as the variables.
            means = means.to(variables[task].device)
            stds = stds.to(variables[task].device)
            # Denormalize the variables.
            denormalized_variables[task] = variables[task] * stds + means
        return denormalized_variables

    def get_normalization_constants(self, task, residuals=False):
        """
        Returns the normalization constants for a given task.

        Parameters
        ----------
        task: str
            The name of the task.
        residuals: bool, optional
            If True, returns the normalization constants for the residuals.
            Otherwise, returns the normalization constants for the location at t=0.

        Returns
        -------
        means: torch.Tensor
        stds: torch.Tensor
        """
        # Retrieve the output variables of the task
        if not residuals:
            if self.output_tabular_tasks[task]["predict_residuals"]:
                time_steps = [0]
            else:
                time_steps = self.target_steps
            variables = self.get_task_output_variables(task, time_steps=time_steps)
        else:   
            variables = self.get_task_output_variables(task, residuals=True)
        # Retrieve the corresponding normalization constants.
        means = torch.tensor(self.tabular_means[variables].values, dtype=torch.float32)
        stds = torch.tensor(self.tabular_stds[variables].values, dtype=torch.float32)
        return means, stds

    def __len__(self):
        return self.trajectories.shape[0]

    def __getitem__(self, idx):
        """
        Returns
        -------
        - input_time_series: Mapping of str to torch.Tensor
            Contextual variables at the past time steps.
            input_time_series[var] is a tensor of shape (P,).
        - input_datacubes: Mapping of str to torch.Tensor
            Input datacubes at the past time steps.
            input_datacubes[name] is a tensor of shape (C, P, H, W).
        - output_time_series: Mapping of str to torch.Tensor
            Target variables at the target time steps.
            output_time_series[task] is a tensor of shape (T,).
        - output_residues: Mapping of str to torch.Tensor
            Residuals between the target variables at each time step and
            the variable at t=0.
        """
        # Retrieve the input time series.
        input_time_series = {}
        for var in self.input_columns:
            cols = [f"{var}_{i}" for i in range(-self.past_steps + 1, 1)]
            input_time_series[var] = torch.tensor(
                self.trajectories[cols].iloc[idx].values, dtype=torch.float32
            )

        # Retrieve the output time series.
        output_time_series = {}
        for task in self.output_tabular_tasks:
            # If the task splits the targets into Y_0 and (Y_T - Y_0), only yield Y_0 here.
            if self.output_tabular_tasks[task]["predict_residuals"]:
                time_steps = [0]
            else:
                # Else, yield all the target steps.
                time_steps = self.target_steps
            output_variables = self.get_task_output_variables(task, time_steps=time_steps)
            output_time_series[task] = torch.tensor(
                self.trajectories[output_variables].iloc[idx].values, dtype=torch.float32
            )
        # Retrieve the residuals for the task that require them.
        output_residues = {}
        for task in self.output_tabular_tasks:
            if not self.output_tabular_tasks[task]["predict_residuals"]:
                continue
            residuals = self.get_task_output_variables(task, residuals=True)
            output_residues[task] = torch.tensor(
                self.trajectories[residuals].iloc[idx].values, dtype=torch.float32
            )

        # Retrieve the index of the sample in the datacubes (i.e. the index of the storm at time t).
        datacube_index = self.trajectories.index[idx]
        # Retrieve the input datacubes.
        input_datacubes = {}
        for name in self.input_datacubes:
            datacube = self.datacubes[name][
                datacube_index - self.past_steps + 1 : datacube_index + 1
            ]
            # Convert the datacube from shape (P, C, H, W) to (C, P, H, W), as expected by torch.
            datacube = datacube.transpose(1, 0)
            # Convert the datacube to a TVTensor, to indicate to torch that the transforms should
            # be applied to it.
            input_datacubes[name] = tv_tensors.Image(datacube)

        # Apply the transforms to the input datacubes.
        if isinstance(self.transforms, list):
            input_datacubes = [transform(input_datacubes) for transform in self.transforms]
        else:
            input_datacubes = self.transforms(input_datacubes)

        return input_time_series, input_datacubes, output_time_series, output_residues

    def get_task_output_variables(self, task, residuals=False, time_steps=None):
        """
        For a given task name, returns the list of the output variables:
        V_t for every t in target_steps for every V in the task's output_variables.

        Parameters
        ----------
        task: str
            The name of the task.
        residuals: bool, optional
            If True, returns the residuals between the target variables and the variable at t=0.
            Otherwise, returns the target variables.
        time_steps: list of int, optional
            List of time steps to consider, which must be a subset of self.target_steps.
        """
        if time_steps is None:
            time_steps = self.target_steps
        return [
            f"{var}_{t}" if not residuals else f"DELTA_{var}_{t}"
            for var in self.output_tabular_tasks[task]["output_variables"]
            for t in time_steps
        ]

    def target_support(self, variable):
        """
        Returns the support of the distribution of a variable in the target,
        as a tuple (min, max).
        """
        return self.trajectories[variable].min(), self.trajectories[variable].max()

    def datacube_shape(self, name):
        """
        Returns the shape of a datacube, as a tuple
        (C, P, H, W).
        """
        # Retrieve the number of channels of the datacube:
        c = self.datacubes[name].shape[1]
        # Retrieve the size of the datacube after cropping:
        h, w = self.patch_size, self.patch_size
        return c, self.past_steps, h, w

    def context_size(self):
        """
        Returns the number of contextual variables.
        """
        return len(self.input_columns) * self.past_steps 

    def get_sample_intensities(self):
        """
        For every sequences S_1, ..., S_N, returns the intensities
        at each time step.
        Can only be called if the task 'vmax' is enabled.

        Returns
        -------
        intensities: torch.Tensor
            The intensities of the samples as tensor of shape (N, T).
        """
        if "vmax" not in self.output_tabular_tasks:
            raise ValueError("The task 'vmax' must be enabled to call get_sample_intensities.")
        vmax_target = self.get_task_output_variables("vmax")
        intensities = torch.tensor(
            self.trajectories[vmax_target].values, dtype=torch.float32
        )
        # Denormalize the intensities
        intensities = self.denormalize_tabular_target({"vmax": intensities})["vmax"]
        return intensities
