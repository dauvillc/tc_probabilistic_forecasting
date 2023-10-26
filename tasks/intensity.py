"""
Clément Dauvilliers - 2023 10 20
Implements the intensity prediction task and metrics.
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
from data_processing import load_ibtracs_data


def intensity_dataset():
    """
    Builds a dataset for the intensity prediction task, with variables
    (storm_id, time, lat, lon, intensity).
    
    Returns
    -------
    dataset : pandas DataFrame
        Dataset for the intensity prediction task.
    """
    # Load the IBTrACS preprocessed data
    ibtracs_data = load_ibtracs_data()

    # ====== Eliminating incomplete tracks ======
    # Retrieve the SIDs of all storms for which the USA_WIND variable is missing
    incomplete_storms = ibtracs_data[ibtracs_data['USA_WIND'].isna()]['SID'].unique()
    # Remove all the rows corresponding to these storms
    ibtracs_data = ibtracs_data[~ibtracs_data['SID'].isin(incomplete_storms)]

    # Rename the USA_WIND column to INTENSITY
    ibtracs_data.rename(columns={'USA_WIND': 'INTENSITY'}, inplace=True)

    # Select the relevant columns
    ibtracs_data = ibtracs_data[['SID', 'ISO_TIME', 'LAT', 'LON', 'INTENSITY']]

    return ibtracs_data.reset_index(drop=True)


def plot_intensity_bias(y_true, y_pred, savepath=None):
    """
    Plots the distribution of the bias of the intensity prediction for each
    predicted time step.
    
    Parameters
    ----------
    y_true : torch Tensor or ndarray of shape (N, n_predicted_steps)
        True intensity values, in m/s.
    y_pred : torch Tensor or ndarray of shape (N, n_predicted_steps)
        Predicted intensity values.
    savepath : str, optional
        Path to save the figure to. If None, the figure is not saved.

    Returns
    -------
    average_bias: float
        Average absolute bias of the intensity prediction, in m/s.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy(force=True)
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy(force=True)

    n_predicted_steps = y_true.shape[1]
    # Compute the bias
    bias = y_pred - y_true
    # Compute the maximum absolute value of the bias, for plotting purposes
    xlim = np.max(np.abs(bias))

    if savepath is not None:
        # Plot the bias distribution in a separate subplot for each predicted time step:
        fig, axes = plt.subplots(nrows=n_predicted_steps, ncols=1, figsize=(10, 5 * n_predicted_steps), squeeze=False)
        for i in range(n_predicted_steps):
            # Plot the distribution of the bias for the i-th predicted time step
            axes[i, 0].hist(bias[:, i], bins=100)
            axes[i, 0].set_xlim(-xlim, xlim)
            axes[i, 0].set_xlabel("Bias (m/s)")
            axes[i, 0].set_ylabel("Frequency")
            axes[i, 0].set_title(f"Bias distribution at t+{i + 1}")

        # Save the figure
        plt.tight_layout()
        plt.savefig(savepath)

    # Return the average bias
    return np.mean(np.abs(bias), axis=0)

def plot_intensity_distribution(y_true, y_pred, savepath=None):
    """
    Plots the distribution of the intensity prediction for each predicted time
    step, versus the true distribution.

    Parameters
    ----------
    y_true : torch Tensor or ndarray of shape (N, n_predicted_steps)
        True intensity values, in m/s.
    y_pred : torch Tensor or ndarray of shape (N, n_predicted_steps)
        Predicted intensity values.
    savepath : str, optional
        Path to save the figure to. If None, the figure is not saved.
    """
    # Plot the distributions of the true and predicted intensities
    # in a separate subplot for each predicted time step:
    n_predicted_steps = y_true.shape[1]
    fig, axes = plt.subplots(nrows=n_predicted_steps, ncols=1, figsize=(10, 5 * n_predicted_steps), squeeze=False)
    for i in range(n_predicted_steps):
        # Plot the distribution of the true and predicted intensities for the i-th predicted time step
        axes[i, 0].hist(y_true[:, i], bins=100, alpha=0.5, label='True')
        axes[i, 0].hist(y_pred[:, i], bins=100, alpha=0.5, label='Predicted')
        axes[i, 0].set_xlabel("Intensity (m/s)")
        axes[i, 0].set_ylabel("Frequency")
        axes[i, 0].set_title(f"Intensity distribution at t+{i + 1}")
        axes[i, 0].legend()
    # Save the figure if a path is provided
    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath)
