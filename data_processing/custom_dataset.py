"""
Clément Dauvilliers - 2023 10 18
Implements functions to convert datacubes between different formats.
"""
import pandas as pd
import torch


class SuccessiveStepsDataset(torch.utils.data.Dataset):
    """
    Dataset of successive steps of storm trajectories and datacubes.

    Given a dataframe of storm trajectories and a datacube, and
    a number of past steps p and future steps n,
    yields tuples (past_traj, past_data, future_traj) where:
        - past_traj is a pandas DataFrame with columns
            (V0_-p+1,V0_-p+2,...,V0_0, V1_-p+1,V1_-p+2,...,V1_0, Vv_-p+1,Vv_-p+2,...,Vv_0)
            where Vi for i in [0, v] are the variables of the trajectories (e.g. lat, lon, intensity, ...).
        - past_data is a torch tensor of dimensions (C, Time, H, W).
        - future_traj is a pandas DataFrame with columns
            (V0_1,V0_2,...,V0_n, V1_1,V1_2,...,V1_n, Vv_1,Vv_2,...,Vv_n)

    Parameters
    ----------
    trajectories : pandas DataFrame
        Dataframe of storm trajectories, with columns (SID, ISO_TIME, V0, V1, ..., Vv).
    datacube : torch tensor of dimensions (sid_time, channels, height, width).
    past_steps : int
        Number of past steps to include in the input.
    future_steps : int
        Number of future steps to include in the output.
    input_variables : list of str, optional
        List of variables to include in the input. If None, all variables are included.
    target_variables : list of str, optional
        List of variables to include in the target. If None, all variables are included.
    target_to_tensor: bool, optional
        If True, converts the target to a torch tensor. The default is True.
    yield_input_variables: bool, optional
        If True (default), yields the past variables alongside the past datacube. If False,
        returns None, input_datacube, target_variables.
    normalise_inputs: bool, optional
        If True, normalises the inputs (datacube and past variable). The default is True.
        The datacube is normalised channel-wise.
    normalise_from: SuccessiveStepsDataset, optional
        If not None, normalises the inputs using the mean and standard deviation of the
        datacube and past variables of the given dataset. The default is None.
    """
    def __init__(self, trajectories, datacube, past_steps, future_steps,
                 input_variables=None, target_variables=None,
                 target_to_tensor=True, yield_input_variables=True,
                 normalise_inputs=True, normalise_from=None):
        # Reset the index of the trajectories to make sure it matches that of
        # the datacube, and assert they have the same length.
        trajectories = trajectories.reset_index(drop=True)
        self.trajectories = trajectories
        assert len(trajectories) == len(datacube), "The trajectories and the datacube have different lengths."
        self.datacube = datacube
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.target_to_tensor = target_to_tensor
        self.yield_input_variables = yield_input_variables
        self.normalise_inputs = normalise_inputs

        # Assert there are no missing values in the trajectories
        assert not trajectories.isna().any().any(), "There are missing values in the trajectories."

        # Retrieve the names of all variables in the trajectories
        self.variables = [col for col in trajectories.columns if col not in ['SID', 'ISO_TIME']]
        self.input_variables = self.variables if input_variables is None else input_variables
        self.target_variables = self.variables if target_variables is None else target_variables

        # If normalise_inputs is True, normalise the datacube and the past variables
        if normalise_inputs:
            if normalise_from is None:
                # Compute the mean and standard deviation of each channel of the datacube,
                # which has shape (sid_time, height, width, channels)
                self.input_datacube_mean = self.datacube.mean(dim=(0, 1, 2))
                self.input_datacube_std = self.datacube.std(dim=(0, 1, 2))
                # Compute the mean and standard deviation of each input variable
                self.input_variables_mean = trajectories[self.input_variables].mean()
                self.input_variables_std = trajectories[self.input_variables].std()
            else:
                # Use the mean and standard deviation of the given dataset
                self.input_datacube_mean = normalise_from.input_datacube_mean
                self.input_datacube_std = normalise_from.input_datacube_std
                self.input_variables_mean = normalise_from.input_variables_mean
                self.input_variables_std = normalise_from.input_variables_std
            # Normalise the datacube
            self.datacube = (self.datacube - self.input_datacube_mean) / self.input_datacube_std
            # Normalise the input variables
            trajectories[self.input_variables] -= self.input_variables_mean
            trajectories[self.input_variables] /= self.input_variables_std

        # Create an empty dataframe to store the past trajectories
        past_trajs = pd.DataFrame({'SID': trajectories['SID'], 'ISO_TIME': trajectories['ISO_TIME']})
        # For every input variable, add a column for each past step
        for var in self.input_variables:
            for i in range(past_steps - 1, -1, -1):
                past_trajs[f'{var}_{-i}'] = trajectories.groupby('SID')[var].shift(i)
        # The first past_steps rows of each storm contain NaN values, as there are not enough past steps
        past_trajs = past_trajs.dropna()

        # Do the same for the future trajectories 
        future_trajs = pd.DataFrame({'SID': trajectories['SID'], 'ISO_TIME': trajectories['ISO_TIME']})
        for var in self.target_variables:
            for i in range(1, future_steps + 1):
                future_trajs[f'{var}_{i}'] = trajectories.groupby('SID')[var].shift(-i)
        # The last future_steps rows of each storm contain NaN values, as there are not enough future steps
        future_trajs = future_trajs.dropna()

        # Join the past and future trajectories on their index, and NOT on the SID and ISO_TIME columns.
        # This is to conserve the matching between the trajectories and the datacube's indices.
        self.target = past_trajs.join(future_trajs.drop(columns=['SID', 'ISO_TIME']),
                                           how='inner', sort=True)
        # Drop the SID and ISO_TIME columns from the target, as we don't want to yield them
        # along with the variables.
        self.target = self.target.drop(columns=['SID', 'ISO_TIME'])
        # Divide the target into past and future trajectories
        self.past_target = self.target[[f'{var}_{i}' for var in input_variables
                                                     for i in range(-past_steps + 1, 1)]]
        self.future_target = self.target[[f'{var}_{i}' for var in target_variables
                                                       for i in range(1, future_steps + 1)]]

        # At this point, the index of target correspond to the index of same SID and ISO_TIME in the
        # datacube. To avoid losing this matching (due to an operation or a reset_index), we'll save that
        # index separately.
        self.target_index = self.target.index.values

        # Convert the target to a torch tensor if needed
        if target_to_tensor:
            self.past_target = torch.tensor(self.past_target.to_numpy(), dtype=torch.float32)
            self.future_target = torch.tensor(self.future_target.to_numpy(), dtype=torch.float32)


    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        # Retrieve the past and future trajectories corresponding to the index
        if self.target_to_tensor: 
            past_traj = self.past_target[idx]
            future_traj = self.future_target[idx]
        else:
            past_traj = self.past_target.iloc[idx]
            future_traj = self.future_target.iloc[idx]
        # Build the indices in the datacube that correspond to the past steps
        past_indices = [self.target_index[idx] - i for i in range(self.past_steps + 1, 1, -1)]
        # Select the datacube samples corresponding to the past steps
        past_data = self.datacube[past_indices]
        # Transpose the datacube to (C, Time, H, W) to be compatible with the input of a torch CNN
        past_data = past_data.transpose(0, 1)
        
        if self.yield_input_variables:
            return past_traj, past_data, future_traj
        else:
            return None, past_data, future_traj

    def target_support(self, variable):
        """
        Returns the support of the distribution of a variable in the target,
        as a tuple (min, max).
        """
        return self.trajectories[variable].min(), self.trajectories[variable].max()

    def patch_size(self):
        """
        Returns the size of the datacube patches.
        """
        return self.datacube.shape[2:]
    
    def datacube_channels(self):
        """
        Returns the number of channels in the datacube.
        """
        return self.datacube.shape[1]

