"""
Evaluates a set of M models, where each model is evaluated on K folds.
"""

import sys

sys.path.append("./")
import argparse
import wandb
import pandas as pd
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils.utils import sshs_category
from utils.wandb import retrieve_wandb_runs
from utils.io import load_predictions_and_targets
from utils.utils import matplotlib_markers


if __name__ == "__main__":
    # =================== ARGUMENTS PARSING =================== #
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the evaluation run.")
    parser.add_argument(
        "-g",
        "--groups",
        nargs="+",
        help="W&B groups to evaluate. Each group should be an ensemble of runs\
                        of the same model on different folds.",
    )
    parser.add_argument(
        "-i",
        "--ids",
        nargs="+",
        type=str,
        help="List of run ids to evaluate. Ignored if groups is provided.",
    )
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        default="CRPS",
        help="Name of the metric to evaluate. Default is CRPS.",
    )
    parser.add_argument(
            "-t",
            "--task",
            type=str,
            default="vmax",
            help="Name of the task to evaluate. Default is vmax.",
    )
    args = parser.parse_args()
    if args.groups is not None:
        args.ids = None
        groups = args.groups
    else:
        groups = None
    metric = args.metric
    eval_task = args.task

    # Load the path configuration
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    # =================== RUNS CONFIGURATION =================== #
    # Initialize W&B
    current_run_name = args.name
    current_run = wandb.init(
        project="tc_prediction",
        name=current_run_name,
        job_type="eval",
        dir=config["paths"]["wandb_logs"],
    )
    # Retrieve the config of each run
    runs, configs, all_tasks = retrieve_wandb_runs(run_ids=args.ids, group=groups)
    run_ids = list(runs.keys())
    # Retrieve the run name for each run id
    run_names = {run_id: configs[run_id]["experiment"]["name"] for run_id in run_ids}

    # =================== DATA LOADING =================== #
    # Load the predictions and targets for every run
    all_predictions, all_targets = load_predictions_and_targets(run_ids)

    # ================== PREPARATION =================== #
    # Create a directory to store the results
    results_dir = Path("results") / current_run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # =================== METRIC COMPUTATION =================== #
    # First: some metrics require an additional aggregation step
    # E.g. the RMSE requires to take the square root of the MSE
    # For these cases, we'll compute the non-aggregated metric and then
    # apply the aggregation step.
    original_metric = metric
    if metric == "RMSE":
        metric = "MSE"
    # We'll compute the metric for each run, on each fold.
    # We'll store it in a DataFrame with the following columns:
    # run_id, group, fold, time step, category, metric
    # Lists to store the columns of the DataFrame
    col_run_id, col_group, col_fold, col_crps = [], [], [], []
    col_time, col_category = [], []
    for run_id in run_ids:
        # If the task is not performed by the run, skip it
        if eval_task not in all_tasks[run_id]:
            continue
        # Compute the metric for the run using the metric function
        # of the run's PredictionDistribution object
        distrib = all_tasks[run_id][eval_task]["distrib_obj"]
        predictions = all_predictions[run_id][eval_task]
        targets = all_targets[run_id][eval_task]
        # Do not compute the average: obtain the value for each sample and each time step
        values = distrib.metrics[metric](predictions, targets, reduce_mean="none")  # (N, T)
        N, T = values.shape
        # Compute the SSHS category for each target
        vmax_targets = all_targets[run_id]["vmax"]
        targets_cat = sshs_category(vmax_targets.view(-1)).view(N, T)
        # Store the results in the future columns of the DataFrame
        col_run_id += [run_id] * N * T
        col_group += [runs[run_id].group] * N * T
        col_fold += [configs[run_id]["experiment"]["fold"]] * N * T
        for i, t in enumerate(configs[run_id]["experiment"]["target_steps"]):
            col_crps += values[:, i].tolist()
            col_category += targets_cat[:, i].tolist()
            col_time += [6 * t] * N  # 1 time step = 6 hours
    # Assemble the DataFrame
    results = pd.DataFrame(
        {
            "run_id": col_run_id,
            "group": col_group,
            "fold": col_fold,
            "time_step": col_time,
            "category": col_category,
            metric: col_crps,
        }
    )
    # Compute the mean and the std of the metric for each group
    # to obtain a DF (group, mean_metric, std_metric)
    group_results = results.groupby("group").agg({metric: ["mean", "std"]})
    group_results.columns = group_results.columns.droplevel()
    group_results.columns = [f"mean_{metric}", f"std_{metric}"]
    group_results = group_results.reset_index()
    # Save the results to a CSV file
    results.to_csv(results_dir / "results.csv", index=False)
    
    # ================== OPTIONAL AGGREGATION =================== #
    # If the metric requires an aggregation step, apply it now
    # We'll create two now DFs: one with the results aggregated by time step
    # and another one with the results aggregated by SSHS category
    if original_metric == "RMSE":
        results_per_time = results.rename(columns={"MSE": "RMSE"})
        results_per_time = results_per_time[['group', 'time_step', 'RMSE']]
        results_per_time = results_per_time.groupby(["group", "time_step"])['RMSE'].mean().reset_index()
        results_per_time['RMSE'] = np.sqrt(results_per_time['RMSE'])
        results_per_cat = results.rename(columns={"MSE": "RMSE"})
        results_per_cat = results_per_cat[['group', 'category', 'RMSE']]
        results_per_cat = results_per_cat.groupby(['group', "category"])['RMSE'].mean().reset_index()
        results_per_cat['RMSE'] = np.sqrt(results_per_cat['RMSE'])
        metric = "RMSE"
    else:
        results_per_time = results
        results_per_cat = results
    # Save the aggregated results to a CSV file
    results_per_time.to_csv(results_dir / "results_per_time.csv", index=False)
    results_per_cat.to_csv(results_dir / "results_per_cat.csv", index=False)

    # ================== PLOTTING =================== #
    sns.set_theme(style="whitegrid", context="poster")
    # Set the legend box to fancy and add a shadow
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.shadow"] = True
    def place_legend(ax):
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.23),
                  ncol=len(run_ids))
    # Create a list of markers and colors for the plots. Each run will be represented
    # by a the same marker and color across all plots.
    names = [configs[run_id]['experiment']['name'] for run_id in run_ids]
    markers = matplotlib_markers(len(run_ids))
    markers = {n: marker for n, marker in zip(names, markers)}
    cmap = plt.get_cmap('tab10', len(run_ids))
    colors = {n: cmap(k) for k, n in enumerate(names)}

    # Plot the metric for each time step, grouped by group
    fig, ax = plt.subplots()
    sns.boxplot(data=results_per_time, x="time_step", y=metric, hue="group", ax=ax,
                palette=colors)
    ax.set_title(f"{eval_task} - {metric} over lead time")
    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel(metric)
    # Set the x-axis to be in hours
    ax.set_xticks(range(len(results_per_time["time_step"].unique())))
    ax.set_xticklabels([f"{t}h" for t in results_per_time["time_step"].unique()])
    # Put the legend below the plot
    place_legend(ax)
    plt.savefig(results_dir / f"{eval_task}_{metric}_lead_time.png",
                bbox_inches='tight')
    # Do the same but with a line plot
    fig, ax = plt.subplots()
    sns.lineplot(data=results_per_time, x="time_step", y=metric, hue="group", ax=ax,
                 palette=colors, markers=markers)
    ax.set_title(f"{eval_task} - {metric} over lead time")
    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel(f'{metric} - 95% CI')
    ax.set_xticks(results_per_time["time_step"].unique())
    ax.set_xticklabels([f"{t}h" for t in results_per_time["time_step"].unique()])
    place_legend(ax)
    plt.savefig(results_dir / f"{eval_task}_{metric}_lead_time_line.png",
                bbox_inches='tight')

    # Plot the metric for each SSHS category, grouped by group
    fig, ax = plt.subplots()
    sns.boxplot(data=results_per_cat, x="category", y=metric, hue="group", ax=ax,
                palette=colors)
    ax.set_title(f"{eval_task} - {metric} over SSHS category")
    ax.set_xlabel("SSHS category")
    ax.set_ylabel(metric)
    ax.set_xticks(range(-1, 6))
    ax.set_xticklabels(["TD", "TS", "C1", "C2", "C3", "C4", "C5"])
    place_legend(ax)
    plt.savefig(results_dir / f"{eval_task}_{metric}_category.png",
                bbox_inches='tight')
    # Do the same but with a line plot
    fig, ax = plt.subplots()
    sns.lineplot(data=results_per_cat, x="category", y=metric, hue="group", ax=ax,
                 palette=colors, markers=markers)
    ax.set_title(f"{eval_task} - {metric} over SSHS category")
    ax.set_xlabel("SSHS category")
    ax.set_ylabel(f'{metric} - 95% CI')
    ax.set_xticks(range(-1, 6))
    ax.set_xticklabels(["TD", "TS", "C1", "C2", "C3", "C4", "C5"])
    place_legend(ax)
    plt.savefig(results_dir / f"{eval_task}_{metric}_category_line.png",
                bbox_inches='tight')
