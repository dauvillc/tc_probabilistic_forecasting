"""
Implements the SuccessiveStepsDataset class, which is a subclass of the
torch.utils.data.Dataset class. This class is used to yield successive
steps of a multiple time series, which can be either tabular data or images.
"""

import torch
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
    random_rotations: bool, optional
        If True, the input datacubes are randomly rotated by an angle between -180 and 179 degrees.
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
        random_rotations=False,
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
        self.random_rotations = random_rotations

        if self.random_rotations:
            # Create the random transformations to apply to the datacubes.
            # The datacube is randomly rotated by angle between -180 and 179 degrees,
            # and is then center-cropped to 64x64 pixels.
            self.transforms = v2.Compose(
                [
                    v2.RandomRotation(degrees=(-180, 179)),
                    v2.CenterCrop(cfg["experiment"]["patch_size"]),
                ]
            )
        else:
            # Otherwise, just crop the center of the datacube.
            self.transforms = v2.Compose([v2.CenterCrop(cfg["experiment"]["patch_size"])])

    def denormalize_tabular_target(self, variables):
        """
        Denormalize a batch of target variables, using the constants stored in the dataset.

        Parameters
        ----------
        variables: Mapping of str to torch.Tensor
            The variables to denormalize. The keys are tasks names and the values are
            torch.Tensors.

        Returns
        -------
        denormalized_variables: Mapping of str to torch.Tensor
            Structure identical to variables, but with the denormalized values.
        """
        denormalized_variables = {}
        for task in variables:
            # Retrieve the normalization constants for the task and convert them to tensors.
            means, stds = self.get_normalization_constants(task)
            # Load them to the same device as the variables.
            means = means.to(variables[task].device)
            stds = stds.to(variables[task].device)
            # Denormalize the variables.
            denormalized_variables[task] = variables[task] * stds + means
        return denormalized_variables

    def get_normalization_constants(self, task):
        """
        Returns the normalization constants for a given task.

        Parameters
        ----------
        task: str
            The name of the task.

        Returns
        -------
        means: torch.Tensor
        stds: torch.Tensor
        """
        # Retrieve the output variables of the task
        variables = self.get_task_output_variables(task)
        # Each variable has the form VAR_t, where t is the time step.
        # The constants to use are then those of VAR.
        # Note: the variable name can contain a '_' character.
        variables = ["_".join(var.split("_")[:-1]) for var in variables]
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
            # Group the output variables of the task into a single tensor.
            output_variables = self.get_task_output_variables(task)
            output_time_series[task] = torch.tensor(
                self.trajectories[output_variables].iloc[idx].values, dtype=torch.float32
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
        input_datacubes = self.transforms(input_datacubes)

        return input_time_series, input_datacubes, output_time_series

    def get_task_output_variables(self, task):
        """
        For a given task name, returns the list of the output variables:
        V_t for every t in target_steps for every V in the task's output_variables.
        """
        return [
            f"{var}_{t}"
            for var in self.output_tabular_tasks[task]["output_variables"]
            for t in self.target_steps
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
        h, w = self.transforms.transforms[-1].size
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
            self.output_trajectories[vmax_target].values, dtype=torch.float32
        )
        # Denormalize the intensities
        intensities = self.denormalize_tabular_target({"vmax": intensities})["vmax"]
        return intensities
