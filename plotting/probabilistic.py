"""
Implements functions to plot the performances of probabilistic models.
"""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_mae_per_threshold(mae_per_threshold, y_true_quantiles=None,
                           save_path=None):
    """
    Plots the MAE per threshold.

    Parameters
    ----------
    mae_per_threshold: Pandas DataFrame
        The DataFrame returned by metrics.probabilistic.mae_per_threshold.
    y_true_quantiles: list of floats, optional
        If not None, list of quantiles that define subsets of the data. The MAE per threshold
        is then computed for each subset of the data.
        For example, [0.9] will only compute the MAE per threshold for the 10% most extreme values.
    save_path: str, optional
        If not None, the figure is saved at the given path.
    """
    with sns.axes_style("whitegrid"):
        # Plot the MAE and the MAE per quantile in the same subplot
        # Remark: the general MAE is the MAE for the 0 quantile (all points are considered)
        # Use an increasing color palette
        palette = sns.color_palette("crest", len(y_true_quantiles) + 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(mae_per_threshold["threshold"], mae_per_threshold["MAE"], color=palette[0], marker="s")
        for i, q in enumerate(y_true_quantiles):
            ax.plot(mae_per_threshold["threshold"], mae_per_threshold["MAE_" + str(q)],
                    color=palette[i+1], marker="s")
        ax.set_xlabel("CDF Quantile u")
        ax.set_ylabel("$|y-F^{-1}(u)|$ (m/s)")
        ax.set_title("Distance between true value and prediction at quantile u")
        ax.legend(["MAE"] + [f"MAE (y >= {q})" for q in y_true_quantiles])
        if save_path is not None:
            plt.savefig(save_path)
    return fig
