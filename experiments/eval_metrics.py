"""
Given K models M_1, ..., M_K which perform possibly different tasks,
retrieves the tasks that are common to all models. A distribution
is associated with each task, and each distribution has a set of
metrics.
For every pair (task, distribution), this script plots the metrics
for each model, on the same plot.
"""

import sys

sys.path.append("./")
import os
import argparse
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from collections import defaultdict
from utils.wandb import make_predictions
from utils.utils import matplotlib_markers, sshs_category


if __name__ == "__main__":
    # Set sns style
    sns.set_style("whitegrid")

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--ids",
        required=True,
        nargs="+",
        help="The ids of the experiment to evaluate.",
    )
    parser.add_argument(
        "-p",
        "--param",
        type=str,
        default=None,
        help="Name of a parameter that differs between the runs.\
                                If provided, the names of the runs will be displayed\
                                as the value of this parameter.\
                                Format: 'section.parameter', e.g.\
                                'training_settings.initial_lr'.",
    )
    parser.add_argument(
        "--param_notation",
        type=str,
        default=None,
        help="Notation to use for the main parameter",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        required=True,
        help="Name of the evaluation run.",
    )
    args = parser.parse_args()

    # Initialize W&B
    current_run_name = args.name
    current_run = wandb.init(project="tc_prediction", name=current_run_name, job_type="eval")

    # Make predictions using the models from the runs
    all_runs_configs, all_runs_tasks, all_runs_predictions, targets = make_predictions(
        args.ids, current_run
    )
    # If a parameter was provided, use it to display the run names
    if args.param is not None:
        section, param = args.param.split(".")
        param_notation = args.param_notation if args.param_notation is not None else param
        # Retrieve the value of the parameter for each run
        run_names = {
            run_id: f"{param_notation}={all_runs_configs[run_id][section][param]}"
            for run_id in args.ids
        }
    else:
        # Retrieve the run name for each run id
        run_names = {run_id: all_runs_configs[run_id]["experiment"]["name"] for run_id in args.ids}

    # Create a folder to store the plots
    save_folder = f"figures/evaluation/{current_run_name}"
    os.makedirs(save_folder, exist_ok=True)

    # Create a list of markers and colors for the plots. Each run will be represented
    # by a the same marker and color across all plots.
    markers = matplotlib_markers(len(args.ids))
    markers = {run_id: marker for run_id, marker in zip(args.ids, markers)}
    cmap = plt.get_cmap("tab10", len(args.ids))
    colors = {run_id: cmap(k) for k, run_id in enumerate(args.ids)}

    # ======== GENERAL METRICS ======== #
    # In this section we'll plot the metrics for every task, over the whole dataset.
    # When several models perform the same task, we'll plot them on the same plot.
    # First, we need to retrieve the list of all tasks that are performed by at least
    # one model.
    all_tasks = set()
    for tasks in all_runs_tasks.values():
        all_tasks.update(list(tasks.keys()))
    all_tasks = list(all_tasks)

    # For a given run and a given task, a specific distribution was used for the predictions.
    # Each distributions has a set of metrics associated with it. A metric might be common
    # to several distributions, but not necessarily.
    # We'll build a map M(task_name, metric_name) such that M[t][m] is the list of run ids that
    # implement the metric m for the task t.
    metrics_map = defaultdict(list)
    for task_name in all_tasks:
        # Browse the runs
        # For each run, if the task is implemented, retrieve the metrics and store the run id
        # in the map.
        for run_id in args.ids:
            run_tasks = all_runs_tasks[run_id]
            if task_name in run_tasks:
                # Retrieve the metrics for the task
                metrics = run_tasks[task_name]["distrib_obj"].metrics
                # metrics is a mapping metric_name --> function
                for metric_name in metrics:
                    # Store the run id in the map
                    metrics_map[(task_name, metric_name)].append(run_id)

    # Now for each pair (task, metric), we'll plot the metric for each run that implements it.
    for (task_name, metric_name), run_ids in metrics_map.items():
        # Create a figure with a single plot
        fig, ax = plt.subplots(1, 1)
        results = {}
        for k, run_id in enumerate(run_ids):
            # Retrieve the predictions and targets for the run
            predictions = all_runs_predictions[run_id][task_name]
            task_targets = targets[task_name]
            # Compute the metric without reducing the result to its mean
            metric_fn = all_runs_tasks[run_id][task_name]["distrib_obj"].metrics[metric_name]
            metric_value = metric_fn(predictions, task_targets, reduce_mean=False)
            # Store the results
            results[run_id] = metric_value
        # Make a boxplot of the results, with the run names as xticks
        ax.boxplot(
            [results[run_id] for run_id in run_ids],
            labels=[run_names[run_id] for run_id in run_ids],
            showmeans=True,
            meanline=True,
            meanprops={"color": "red"},
        )
        ax.set_title(f"{task_name} - {metric_name}")
        ax.set_ylabel(metric_name)
        # Add a legend to indicate that the mean is in red and the median in orange
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color="red", lw=1, ls="--", label="Mean"),
                plt.Line2D([0], [0], color="orange", lw=1, label="Median"),
            ]
        )
        # Save the figure
        fig.savefig(os.path.join(save_folder, f"{task_name}-{metric_name}.png"))
        # Log the figure to W&B
        current_run.log({f"{task_name}-{metric_name}": wandb.Image(fig)})
        plt.close(fig)

    # ======== CATEGORY-WISE METRICS ======== #
    # In this section, we'll plot the metrics for every model, for each category of the SSHS.
    # Since we are forecasting multiple time steps ahead, we'll use the maximum SSHS category
    # reached over all time steps.
    # First, we can compute the mask that indicates the SSHS category for each sample.
    target_sshs = sshs_category(targets["vmax"])
    target_sshs, _ = target_sshs.max(dim=1)
    # Now, we'll evaluate every (task, metric) pair.
    for (task_name, metric_name), run_ids in metrics_map.items():
        # Create a figure with a single plot. The X axis will be the SSHS category,
        # and the Y axis the metric value.
        # Each model will be represented by its own line, with confidence intervals.
        fig, ax = plt.subplots(1, 1)
        # Create lists to store the results: model, category, metric value
        # Those will be assembled into a long-format dataframe
        res_ids, res_cats, res_values = [], [], []
        for k, run_id in enumerate(run_ids):
            # Retrieve the predictions and targets for the run
            predictions = all_runs_predictions[run_id][task_name]
            task_targets = targets[task_name]
            # Retrieve the metric function
            metric_fn = all_runs_tasks[run_id][task_name]["distrib_obj"].metrics[metric_name]
            for category in range(-1, 6):
                mask = target_sshs == category
                cat_targets = task_targets[mask]
                # The predictions can be either a tensor or a tuple of tensors
                if isinstance(predictions, tuple):
                    cat_preds = tuple(pred[mask] for pred in predictions)
                else:
                    cat_preds = predictions[mask]
                # Compute the metric for the category
                metric_value = metric_fn(cat_preds, cat_targets, reduce_mean=False)
                # Convert the metric values from tensors to lists
                # If the metric is a scalar tensor, convert it to a list
                metric_value = metric_value.tolist()
                if not isinstance(metric_value, list):
                    metric_value = [metric_value]
                # Store the results
                res_ids = res_ids + [run_id] * len(metric_value)
                res_cats = res_cats + [category] * len(metric_value)
                res_values = res_values + metric_value

        # Assemble the results in a dataframe
        df = pd.DataFrame(
            {
                "id": res_ids,
                "model": [run_names[run_id] for run_id in res_ids],
                "category": res_cats,
                metric_name: res_values,
            }
        )
        # Plot the results
        sns.pointplot(
            data=df,
            x="category",
            y=metric_name,
            hue="model",
            errorbar=("ci", 95),
            dodge=0.2 if df["model"].nunique() > 1 else False,
            linewidth=1.5,
            markersize=5,
            err_kws={"linewidth": 1.5, "markersize": 5},
            ax=ax,
            palette=[colors[run_id] for run_id in df["id"].unique()],
            marker=[markers[run_id] for run_id in df["id"].unique()],
        )
        ax.set_title(f"{task_name} - {metric_name}")
        ax.set_ylabel(f"{metric_name} - 95% CI")
        ax.set_xlabel("Maximum SSHS category over t+6h,12h,18h,24h")
        # Save the figure
        fig.savefig(os.path.join(save_folder, f"{task_name}-{metric_name}-categories.png"))
        current_run.log({f"{task_name}-{metric_name}-categories": wandb.Image(fig)})

    # Plot the number of samples per SSHS category
    # (which is the same for all models)
    fig, ax = plt.subplots(1, 1)
    # Note: bincount expects a tensor of positive integers.
    counts = (target_sshs.to(torch.int) + 1).bincount(minlength=7).tolist()
    ax.bar(range(-1, 6), counts)
    # Write the counts on top of the bars
    for i, count in enumerate(counts):
        ax.text(i - 1, count, str(count), ha="center", va="bottom")
    ax.set_title("Number of samples per SSHS category")
    ax.set_xlabel("Maximum SSHS category over t+6h,12h,18h,24h")
    ax.set_ylabel("Number of samples")
    # Save the figure
    fig.savefig(os.path.join(save_folder, "sshs_counts.png"))
    # Log the figure to W&B
    current_run.log({"sshs_counts": wandb.Image(fig)})
    plt.close(fig)
