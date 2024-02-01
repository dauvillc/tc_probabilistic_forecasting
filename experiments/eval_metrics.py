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
from collections import defaultdict
from utils.wandb import make_predictions
from utils.utils import matplotlib_markers


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ids", required=True, nargs='+',
                        help="The ids of the experiment to evaluate.")
    args = parser.parse_args()

    # Initialize W&B 
    current_run_name = "metrics-" + "-".join(args.ids)
    current_run = wandb.init(project="tc_prediction",
                             name=current_run_name,
                             job_type="eval")

    # Make predictions using the models from the runs
    all_runs_configs, all_runs_tasks, all_runs_predictions, targets = make_predictions(args.ids, current_run)
    # Retrieve the run name for each run id
    run_names = {run_id: all_runs_configs[run_id]['experiment']['name'] for run_id in args.ids}

    # Create a folder to store the plots
    save_folder = f"figures/evaluation/{current_run_name}"
    os.makedirs(save_folder, exist_ok=True)

    # Create a list of markers and colors for the plots. Each run will be represented 
    # by a the same marker and color across all plots.
    markers = matplotlib_markers(len(args.ids))
    markers = {run_id: marker for run_id, marker in zip(args.ids, markers)}
    cmap = plt.get_cmap('tab10', len(args.ids))
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
                metrics = run_tasks[task_name]['distrib_obj'].metrics
                # metrics is a mapping metric_name --> function
                for metric_name in metrics:
                    # Store the run id in the map
                    metrics_map[(task_name, metric_name)].append(run_id)

    # Now for each pair (task, metric), we'll plot the metric for each run that implements it.
    for (task_name, metric_name), run_ids in metrics_map.items():
        # Create a figure with a single plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        results = {}
        for k, run_id in enumerate(run_ids):
            # Retrieve the predictions and targets for the run
            predictions = all_runs_predictions[run_id][task_name]
            task_targets = targets[task_name]
            # Compute the metric without reducing the result to its mean
            metric_fn = all_runs_tasks[run_id][task_name]['distrib_obj'].metrics[metric_name]
            metric_value = metric_fn(predictions, task_targets, reduce_mean=False)
            # Store the results
            results[run_id] = metric_value
        # Make a boxplot of the results, with the run names as xticks
        ax.boxplot([results[run_id] for run_id in run_ids],
                   labels=[run_names[run_id] for run_id in run_ids],
                   showmeans=True,
                   meanline=True,
                   meanprops={'color': 'red'})
        ax.set_title(f"{task_name} - {metric_name}")
        ax.set_ylabel(metric_name)
        # Save the figure
        fig.savefig(os.path.join(save_folder, f"{task_name}-{metric_name}.png"))
        # Log the figure to W&B
        current_run.log({f"{task_name}-{metric_name}": wandb.Image(fig)})
        plt.close(fig)

