"""
Defines functions to plot metrics for deterministic models.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from utils.utils import matplotlib_markers


def plot_deterministic_metrics(predictions, targets, title, unit, save_path=None):
    """
    Plots the metrics a set of deterministic models.
    
    Parameters
    ----------
    predictions : Mapping of str -> torch.Tensor
        The predictions of the models. The keys are model names, and the values
        are the predictions, of shape (N, T, V) where N is the number of samples,
        T the lead time and V the number of variables.
    targets: torch.Tensor of shape (N, T) or (N, T, V)
    title: str
        The title of the figure.
    unit: str
        The unit of the variable.
    save_path : str, optional
        If not None, the figure will be saved at this location.
    
    Returns
    -------
    The figure.
    """
    # First, we need to compute the metrics for each model
    rmse, mae = {}, {}
    for name, pred in predictions.items():
        # Extend the target to (N, T, 1) if they are of shape (N, T)
        if targets.ndim == 2:
            targets = targets.unsqueeze(-1)
        rmse[name] = F.mse_loss(pred, targets, reduction='none').mean(dim=(0, 2)).sqrt()
        mae[name] = F.l1_loss(pred, targets, reduction='none').mean(dim=(0, 2))
    # The figure will contain two subplots, one for the RMSE and one for the MAE
    # The x-axis will be the lead time, and the y-axis the metric value
    # The lines and markers will indicate the model
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
    markers = matplotlib_markers(len(predictions))
    time_steps = np.arange(1, targets.shape[1] + 1) * 6
    # Plot the RMSE
    for i, (name, rmse_) in enumerate(rmse.items()):
        axes[0].plot(time_steps, rmse_, label=name, marker=markers[i])
    axes[0].set_ylabel(f'RMSE ({unit})')
    axes[0].legend()
    # Plot the MAE
    for i, (name, mae_) in enumerate(mae.items()):
        axes[1].plot(time_steps, mae_, label=name, marker=markers[i])
    axes[1].set_ylabel(f'MAE ({unit})')
    axes[1].legend()
    # Set the x-axis label
    axes[1].set_xlabel('Lead time (hours')
    axes[1].set_xticks(time_steps)
    axes[1].set_xticklabels([f'+{t:.0f}' for t in time_steps])
    # Set the title
    axes[0].set_title(title)
    # Save the figure if needed
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    return fig
