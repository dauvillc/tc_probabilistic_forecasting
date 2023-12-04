"""
Implements functions to plot the accuracy of a predicted distribution versus the
true distribution.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from utils.utils import to_numpy


def plot_data_distribution(y, quantiles=None, savepath=None):
    """
    Plots the distribution of a dataset.

    Parameters
    ----------
    y: array-like or torch.Tensor, of shape (N, T)
        where N is the number of samples and T is the number of time steps.
    quantiles: list of floats, optional
        If not None, list of quantiles to plot as vertical lines.
    savepath: str, optional
    """
    y = to_numpy(y)
    with sns.axes_style("whitegrid"):
        # Create a DataFrame with a column for each time step
        df = pd.DataFrame(y, columns=[f"t={t}" for t in range(y.shape[1])])
        # Reshape it to long format
        df = df.melt(var_name="Time step", value_name="wind_speed")
        # Plot the distributions of each time step
        fig, ax = plt.subplots()
        sns.histplot(data=df, x="wind_speed", hue="Time step", ax=ax,
                     kde=True, bins=np.arange(0, 100, 1))

        # Plot the quantiles if provided
        if quantiles is not None:
            for q in quantiles:
                # Plot the quantile as a background vertical line,
                # red and very thin. Write the quantile value next to it.
                ax.axvline(np.quantile(y, q), color="r", alpha=0.75, lw=0.5)
                ax.text(np.quantile(y, q), 0.9 * ax.get_ylim()[1],
                        f"{q:.2f}", rotation=90, ha="right", va="top")
        ax.set_xlabel("1-min Sustained Wind Speed (m/s)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of the true values")
        if savepath is not None:
            fig.savefig(savepath)
        plt.close(fig)

        return fig
