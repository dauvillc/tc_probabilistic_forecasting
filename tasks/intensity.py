"""
Clément Dauvilliers - 2023 10 20
Implements the intensity prediction task and metrics.
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
import pandas as pd
from data_processing.datasets import load_ibtracs_data


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
    Multiple models can be included.
    
    Parameters
    ----------
    y_true : torch Tensor or ndarray of shape (N, n_predicted_steps),
        where N is the number of samples.
        True intensity values, in m/s.
    y_pred : mapping of str to torch Tensor or ndarray of shape (N, n_predicted_steps),
        where N is the number of samples.
    savepath : str, optional
        Path to save the figure to. If None, the figure is not saved.

    Returns
    -------
    average_bias: a mapping of str to ndarray of shape (n_predicted_steps,), such
        that average_bias[model_name][i] is the average absolute bias of the i-th predicted
        time step for the model named model_name.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy(force=True)
    model_names = list(y_pred.keys())
    # Convert the predictions to ndarrays if required
    for model_name in model_names:
        if isinstance(y_pred[model_name], torch.Tensor):
            y_pred[model_name] = y_pred[model_name].numpy(force=True)
    n_predicted_steps = y_true.shape[1]

    # Compute the bias for each model
    bias = {}
    for model_name in model_names:
        bias[model_name] = y_pred[model_name] - y_true

    # Create a DataFrame in long format with (model, time step, bias) as columns
    df = pd.DataFrame(columns=['model', 'time step', 'bias'])
    for model_name in model_names:
        for i in range(n_predicted_steps):
            df = pd.concat([df,
                            pd.DataFrame({'model': [model_name] * len(bias[model_name][:, i]),
                                          'time step': [i + 1] * len(bias[model_name][:, i]),
                                          'bias': bias[model_name][:, i]})],
                           ignore_index=True)

    # Plot the distribution of the bias for each model and for each predicted time step
    # using a violin plot
    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.violinplot(x='time step', y='bias', hue='model', data=df, ax=ax)
        ax.set_xlabel("Predicted time step (hours)")
        ax.set_ylabel("Bias (m/s)")
        ax.set_title("Distribution of the bias of the intensity prediction for each predicted time step")

    # Save the figure if a path is provided
    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath)
    # Compute the average bias for each model and for each predicted time step
    average_bias = {}
    for model_name in model_names:
        average_bias[model_name] = np.mean(np.abs(bias[model_name]), axis=0)
    return average_bias


def plot_intensity_distribution(y_true, y_pred, savepath=None):
    """
    Plots the distribution of the intensity prediction for each predicted time step.
    Multiple models can be included.
    
    Parameters
    ----------
    y_true : torch Tensor or ndarray of shape (N, n_predicted_steps),
        where N is the number of samples.
        True intensity values, in m/s.
    y_pred : mapping of str to torch Tensor or ndarray of shape (N, n_predicted_steps),
        where N is the number of samples.
    savepath : str, optional
        Path to save the figure to. If None, the figure is not saved.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy(force=True)
    model_names = list(y_pred.keys())
    # Convert the predictions to ndarrays if required
    for model_name in model_names:
        if isinstance(y_pred[model_name], torch.Tensor):
            y_pred[model_name] = y_pred[model_name].numpy(force=True)
    n_predicted_steps = y_true.shape[1]
    # Consider the groundtruth as a model
    model_names.append('groundtruth')
    y_pred['groundtruth'] = y_true
    
    # Create a DataFrame in long format with (model, time step, intensity) as columns
    df = pd.DataFrame(columns=['model', 'time step', 'intensity'])
    for model_name in model_names:
        for i in range(n_predicted_steps):
            df = pd.concat([df,
                            pd.DataFrame({'model': [model_name] * len(y_pred[model_name][:, i]),
                                          'time step': [i + 1] * len(y_pred[model_name][:, i]),
                                          'intensity': y_pred[model_name][:, i]})],
                           ignore_index=True)
    # Plot the distribution of the intensity for each model and for each predicted time step
    # using a violin plot
    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.violinplot(x='time step', y='intensity', hue='model', data=df, ax=ax)
        ax.set_xlabel("Predicted time step (hours)")
        ax.set_ylabel("Intensity (m/s)")
        ax.set_title("Distribution of the intensity prediction for each predicted time step")

    # Save the figure if a path is provided
    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath)
