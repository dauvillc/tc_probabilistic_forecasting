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
import numpy as np
from collections import defaultdict
from utils.wandb import retrieve_wandb_runs
from utils.utils import matplotlib_markers, sshs_category
from utils.io import load_predictions_and_targets


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

    # ======== RUNS CONFIGURATION ======== #
    # Initialize W&B
    current_run_name = args.name
    current_run = wandb.init(project="tc_prediction", name=current_run_name, job_type="eval")

    # Retrieve the config of each run
    runs, configs, all_runs_tasks = retrieve_wandb_runs(args.ids)

    # If a parameter was provided, use it to display the run names
    if args.param is not None:
        section, param = args.param.split(".")
        param_notation = args.param_notation if args.param_notation is not None else param
        # Retrieve the value of the parameter for each run
        run_names = {}
        for run_id in args.ids:
            run_config = configs[run_id]
            if param in run_config[section]:
                run_names[run_id] = f"{param_notation}={run_config[section][param]}"
            else:
                run_names[run_id] = f"{param_notation}=None"
    else:
        # Retrieve the run name for each run id
        run_names = {run_id: configs[run_id]["experiment"]["name"] for run_id in args.ids}

    # ======= PLOTTING SETUP ======= #
    # Create a folder to store the plots
    save_folder = f"figures/evaluation/{current_run_name}"
    os.makedirs(save_folder, exist_ok=True)

    # Create a list of markers and colors for the plots. Each run will be represented
    # by a the same marker and color across all plots.
    markers = matplotlib_markers(len(args.ids))
    markers = {run_id: marker for run_id, marker in zip(args.ids, markers)}
    cmap = plt.get_cmap("tab10", len(args.ids))
    colors = {run_id: cmap(k) for k, run_id in enumerate(args.ids)}

    # ======== LOAD THE PREDICTIONS AND TARGETS ======== #
    all_runs_predictions, all_runs_targets = load_predictions_and_targets(args.ids)

    # ======== GENERAL METRICS ======== #
    # In this section we'll plot the metrics for every task, over the whole dataset.
    # When several models perform the same task, we'll plot them on the same plot.
    # First, we need to retrieve the list of all tasks that are performed by at least
    # one model.
    common_tasks = set()
    for task_dict in all_runs_tasks.values():
        common_tasks.update(list(task_dict.keys()))
    common_tasks = list(common_tasks)

    # For a given run and a given task, a specific distribution was used for the predictions.
    # Each distributions has a set of metrics associated with it. A metric might be common
    # to several distributions, but not necessarily.
    # We'll build a map M(task_name, metric_name) such that M[t][m] is the list of run ids that
    # implement the metric m for the task t.
    metrics_map = defaultdict(list)
    for task_name in common_tasks:
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
            task_targets = all_runs_targets[run_id][task_name]
            # Compute the metric without reducing the result to its mean
            metric_fn = all_runs_tasks[run_id][task_name]["distrib_obj"].metrics[metric_name]
            metric_value = metric_fn(predictions, task_targets, reduce_mean="time")
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
            targets = all_runs_targets[run_id]
            task_targets = targets[task_name]
            # First, we can compute the mask that indicates the SSHS category for each sample.
            target_sshs = sshs_category(targets["vmax"])
            target_sshs, _ = target_sshs.max(dim=1)
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
                metric_value = metric_fn(cat_preds, cat_targets, reduce_mean="time")
                # Convert the metric values from tensors to lists
                metric_value = metric_value.tolist()
                if not isinstance(metric_value, list):  # If the metric is a scalar
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
            dodge=0.05 if df["model"].nunique() > 1 else False,
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

    # ==== TIME-STEP WISE METRICS ==== #
    # In this section, we'll plot the metrics for every model, for each time step.
    # We'll use the same approach as for the SSHS categories.
    for (task_name, metric_name), run_ids in metrics_map.items():
        # Create a figure with a single plot. The X axis will be the time step,
        # and the Y axis the metric value.
        # Each model will be represented by its own line, with confidence intervals.
        fig, ax = plt.subplots(1, 1)
        # Create lists to store the results: model, time step, metric value
        # Those will be assembled into a long-format dataframe
        res_ids, res_steps, res_values = [], [], []
        for k, run_id in enumerate(run_ids):
            # Retrieve the predictions and targets for the run
            predictions = all_runs_predictions[run_id][task_name]
            targets = all_runs_targets[run_id]
            task_targets = targets[task_name]
            # Retrieve the metric function
            metric_fn = all_runs_tasks[run_id][task_name]["distrib_obj"].metrics[metric_name]
            # Evaluate the metric by averaging over the samples and not over time
            metric_value = metric_fn(
                predictions, task_targets, reduce_mean="none"
            )  # (N, T) or (N,) or (T,)
            # Retrieve the predicted time steps.
            exp_cfg = configs[run_id]["experiment"]
            T = np.array(exp_cfg["target_steps"]) * 6 # 6h time steps
            for i, t in enumerate(T):
                # Get the scores of all samples at time t
                if metric_value.dim() == 1:
                    if metric_value.shape[0] == len(T):
                        # Metric that has one value for each time step
                        # Consider it constant over time
                        values_at_t = [metric_value[i].item()]
                    else:
                        # Metric that has a single value for all time steps, for each sample
                        values_at_t = metric_value.tolist()
                else:
                    # Metric that has a value for each sample at each time step
                    values_at_t = metric_value[:, i].tolist()
                # Store the results in long format
                res_ids += [run_id] * len(values_at_t)
                res_steps += [t] * len(values_at_t)
                res_values += values_at_t

        # Assemble the results in a dataframe
        df = pd.DataFrame(
            {
                "id": res_ids,
                "model": [run_names[run_id] for run_id in res_ids],
                "step": res_steps,
                metric_name: res_values,
            }
        )
        # Plot the results
        sns.pointplot(
            data=df,
            x="step",
            y=metric_name,
            hue="model",
            errorbar=("ci", 95),
            dodge=0.05 if df["model"].nunique() > 1 else False,
            linewidth=1.5,
            markersize=5,
            err_kws={"linewidth": 1.5, "markersize": 5},
            ax=ax,
            palette=[colors[run_id] for run_id in df["id"].unique()],
            marker=[markers[run_id] for run_id in df["id"].unique()],
        )
        ax.set_title(f"{task_name} - {metric_name}")
        ax.set_ylabel(f"{metric_name} - 95% CI")
        ax.set_xlabel("Time step (hours)")
        # Save the figure
        fig.savefig(os.path.join(save_folder, f"{task_name}-{metric_name}-lead_time.png"))
        current_run.log({f"{task_name}-{metric_name}-lead_time": wandb.Image(fig)})
