"""
Implements functions to plot the results of the multiple quantile loss
regression.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import matplotlib_markers
from metrics.loss_functions import MultipleQuantileLoss


def plot_quantile_losses(y_pred, y_true, quantiles, savepath=None):
    """
    Plots the individual quantile losses for each quantile.

    Parameters
    ----------
    y_pred : mapping of str to torch.Tensor or array-like of shape \
        (n_samples, n_time_steps, n_quantiles).
        Keys are model names, values are the predicted quantiles from the
        model.
    y_true : array-like of shape (n_samples, n_time_steps).
        The true values.
    quantiles : array-like of shape (n_quantiles,).
        The quantiles to plot.
    savepath : str or None, default=None
        If not None, the plot is saved at this path.

    Returns
    -------
    The figure.
    """
    future_steps = y_true.shape[1]
    model_names = list(y_pred.keys())
    losses = {}
    eval_loss_function = MultipleQuantileLoss(quantiles=quantiles, reduction="none", normalize=False)
    for name, preds in y_pred.items():
        losses[name] = eval_loss_function(preds, y_true).mean(dim=0).cpu().numpy()
    # Plot the losses in one subplot per time step, as rows
    with sns.axes_style("whitegrid"):
        # Obtain markers for each model
        markers = matplotlib_markers(len(model_names))
        markers = {name: markers[i] for i, name in enumerate(model_names)}
        fig, axes = plt.subplots(future_steps, 1, figsize=(12, 8))
        for i in range(future_steps):
            ax = axes[i]
            # Plot the losses for each model
            # The losses have shape (n_time_steps, n_quantiles)
            for name, loss in losses.items():
                ax.plot(loss[i], label=name, marker=markers[name])
            ax.set_xlabel("Quantile")
            ax.set_ylabel("Loss (m/s)")
            ax.set_title(f"Loss for time step t+{i+1}")
            ax.set_xticks(range(len(quantiles)))
            # Write the quantiles as labels, rounded to 2 decimals and rotated
            ax.set_xticklabels(np.array(quantiles).round(2), rotation=45)
            ax.legend()
        # Save the figure
        plt.tight_layout()
        plt.savefig(savepath)
    return fig


