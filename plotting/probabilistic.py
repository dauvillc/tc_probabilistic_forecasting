"""
Implements functions to plot the performances of probabilistic models.
"""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metric_per_threshold(metric, metric_per_threshold, y_true_quantiles=None,
                              save_path=None):
    """
    Plots a certain metric as a function of the threshold.

    Parameters
    ----------
    metric: str
        The name of the metric to plot. Can be "bias", "MAE", or "RMSE".
    metric_per_threshold: Pandas DataFrame
        The DataFrame returned by metrics.probabilistic.metric_per_threshold.
    y_true_quantiles: list of floats, optional
        If not None, list of quantiles that define subsets of the data. The metric 
        is then computed for each subset of the data.
        For example, [0.9] will only compute the metric per threshold for the 10% most extreme values.
    save_path: str, optional
        If not None, the figure is saved at the given path.
    """
    # Define the title and y label of the figure according to the metric
    if metric == "bias":
        title = "Bias between true value and predicted value at quantile u"
        ylabel = "$y-F^{-1}(u)$ (m/s)"
    elif metric == "mae":
        title = "Mean absolute error between true value and predicted value at quantile u"
        ylabel = "$|y-F^{-1}(u)|$ (m/s)"
    elif metric == "rmse":
        title = "Root mean squared error between true value and predicted value at quantile u"
        ylabel = "$RMSE(y, F^{-1}(u))$ (m/s)"
    with sns.axes_style("whitegrid"):
        # Plot the metric and the metric per quantile in the same subplot
        # Remark: the general metric is the metric for the 0 quantile (all points are considered)
        # Use an increasing color palette
        palette = sns.color_palette("crest", len(y_true_quantiles) + 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(metric_per_threshold["threshold"], metric_per_threshold[metric], color=palette[0], marker="s")
        for i, q in enumerate(y_true_quantiles):
            ax.plot(metric_per_threshold["threshold"], metric_per_threshold[f"{metric}_" + str(q)],
                    color=palette[i+1], marker="s")
        ax.set_xlabel("CDF Quantile u")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend([metric] + [f"{metric} (y >= {q} quantile)" for q in y_true_quantiles])
        if save_path is not None:
            plt.savefig(save_path)
    return fig
