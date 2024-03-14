"""
Implements the SuccessiveStepsDataset class, which is a subclass of the
torch.utils.data.Dataset class. This class is used to yield successive
steps of a multiple time series, which can be either tabular data or images.
"""

import pandas as pd
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
from utils.utils import sshs_category


class SuccessiveStepsDataset(torch.utils.data.Dataset):
    """
    The class receives a set of time series and returns batches of successive
    steps of it. The time series can be either tabular data or datacubes.
    The class receives a set of input time series, a set of input datacubes,
    a set of output time series and a set of output datacubes.
    If X is an input time series and Y is an output time series, and P is the
    number of past steps and T is the number of future steps, the class returns
    batches of shape (batch_size, P, X.shape[1]) and (batch_size, T, Y.shape[1]).
    If X is an input datacube and Y is an output datacube, the class returns
    batches of shape (batch_size, P, *X.shape[1:]) and (batch_size, T, *Y.shape[1:]).

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
        self.future_steps = cfg["experiment"]["future_steps"]
        self.random_rotations = random_rotations
        # The output variables are the variables that are included in at least one
        # tabular task.
        self.output_columns = set()
        for task in self.output_tabular_tasks:
            output_variables = self.output_tabular_tasks[task]["output_variables"]
            self.output_columns.update(output_variables)
        self.output_columns = list(self.output_columns)

        # Reset the indices of the trajectories dataframe so that trajectories[i] matches
        # the i-th sample in the datacubes.
        self.trajectories.reset_index(inplace=True, drop=True)
        for datacube in self.datacubes.values():
            assert (
                len(self.trajectories) == datacube.shape[0]
            ), "The number of samples in the trajectories dataframe and in the datacubes must match."
        assert not trajectories.isna().any().any(), "There are missing values in the trajectories."

        # Every manipulation of the trajectories will be done separately for each storm.
        grouped_trajs = self.trajectories.groupby("SID")
        # Create a DataFrame for the input time series. For each input variable V_t, it includes
        # the columns V_{-P+1}, ..., V_{0}. We also include the columns SID and ISO_TIME, to make
        # sure input_trajectories has at least one column.
        self.input_trajectories = pd.DataFrame()
        for var in self.input_columns + ["SID", "ISO_TIME"]:
            for i in range(self.past_steps):
                self.input_trajectories[f"{var}_{-i}"] = grouped_trajs[var].shift(i)
        # The first P-1 rows of each storm are NaN since there are no previous steps.
        self.input_trajectories.dropna(inplace=True)

        # Create a DataFrame for the output time series. For each output variable V_t, it includes
        # the columns V_{1}, ..., V_{T}.
        self.output_trajectories = pd.DataFrame()
        for var in self.output_columns + ["SID", "ISO_TIME"]:
            for i in range(1, self.future_steps + 1):
                self.output_trajectories[f"{var}_{i}"] = grouped_trajs[var].shift(-i)
        # The last T rows of each storm are NaN since there are no future steps.
        self.output_trajectories.dropna(inplace=True)

        # Optionally, select only the samples which reach a minimum category over the future steps.
        if (
            "vmax" in self.output_tabular_tasks
            and "train_min_category" in cfg["experiment"]
            and subset == "train"
        ):
            min_category = cfg["experiment"]["train_min_category"]
            # Compute the max intensity of each sample over the future steps.
            intensities = self.get_sample_intensities()  # (N, T)
            max_intensities = intensities.max(dim=1).values
            # Convert the max intensities to categories.
            max_categories = sshs_category(max_intensities)
            # Select the samples which reach at least the minimum category.
            selected_mask = (max_categories >= min_category).numpy()
            self.output_trajectories = self.output_trajectories[selected_mask]

        # Since we removed different rows from the input and output trajs, we need to retain only the intersection
        # of their indices.
        indices = self.input_trajectories.index.intersection(self.output_trajectories.index)
        self.input_trajectories = self.input_trajectories.loc[indices]
        self.output_trajectories = self.output_trajectories.loc[indices]
        self.trajectory_indices = indices
        # Remark: at this point, the indices of self.input_trajectories and self.output_trajectories
        # are the same and match the indices of the datacubes, meaning that self.input_trajectories[i]
        # refers to the same storm and time as datacube[i] for each datacube.
        # However, the trajectories' indices have gaps at the first P rows and the last T rows of each storm.
        # These gaps are not present in the datacubes' indices, on purpose: if input_trajectories[i] refers to
        # a storm at time t, then datacube[i - P + 1] refers to the same storm at time t - P + 1, and datacube[i + T]
        # refers to the same storm at time t + T.
        # We can now remove the columns SID_t and ISO_TIME_t from the input and output time series.
        for i in range(-self.past_steps + 1, 1):
            self.input_trajectories.drop(columns=[f"SID_{i}", f"ISO_TIME_{i}"], inplace=True)
        for i in range(1, self.future_steps + 1):
            self.output_trajectories.drop(columns=[f"SID_{i}", f"ISO_TIME_{i}"], inplace=True)

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
        variables = [var.split("_")[0] for var in variables]
        means = torch.tensor(self.tabular_means[variables].values, dtype=torch.float32)
        stds = torch.tensor(self.tabular_stds[variables].values, dtype=torch.float32)
        return means, stds

    def __len__(self):
        return len(self.trajectory_indices)

    def __getitem__(self, idx):
        """
        Returns the idx-th sample of the dataset, as a tuple
        (input_time_series, input_datacubes, output_time_series).

        input_time_series is a Mapping of str to torch.Tensor, where the keys are the input
        variable names and the values are torch.Tensors of shape (batch_size, P, K) where K
        is the number of variables in the input time series.
        output_time_series is a Mapping of str to torch.Tensor, where the keys are the task names
        and the values are torch.Tensors of shape (batch_size, T, K') where K' is the number of variables
        in the task.

        input_datacubes is a Mapping of str to torch.Tensor, where the keys are the names of the
        input datacubes and the values are torch.Tensors of shape (batch_size, C, P, H, W).
        """
        # Retrieve the input time series.
        input_time_series = {}
        for var in self.input_columns:
            cols = [f"{var}_{i}" for i in range(-self.past_steps + 1, 1)]
            input_time_series[var] = torch.tensor(
                self.input_trajectories[cols].iloc[idx].values, dtype=torch.float32
            )

        # Retrieve the output time series.
        output_time_series = {}
        for task in self.output_tabular_tasks:
            # Group the output variables of the task into a single tensor.
            output_variables = self.get_task_output_variables(task)
            output_time_series[task] = torch.tensor(
                self.output_trajectories[output_variables].iloc[idx].values, dtype=torch.float32
            )

        # Retrieve the index of the sample in the datacubes (i.e. the index of the storm at time t).
        datacube_index = self.trajectory_indices[idx]

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
        For a given task name, returns the list of the output variables
        (VAR1_1, ..., VAR1_T, ..., VARK_1, ..., VARK_T).
        """
        return [
            f"{var}_{i}"
            for var in self.output_tabular_tasks[task]["output_variables"]
            for i in range(1, self.future_steps + 1)
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
        vmax_future_vars = self.get_task_output_variables('vmax')
        intensities = torch.tensor(self.output_trajectories[vmax_future_vars].values, dtype=torch.float32)
        # Denormalize the intensities
        intensities = self.denormalize_tabular_target({"vmax": intensities})[
            "vmax"
        ]
        return intensities
