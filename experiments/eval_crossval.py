"""
Evaluates a set of M models, where each model is evaluated on K folds.
"""

import sys

sys.path.append("./")
import argparse
import wandb
import pandas as pd
import yaml
from pathlib import Path
from utils.wandb import retrieve_wandb_runs
from utils.io import load_predictions_and_targets


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
    args = parser.parse_args()
    if args.groups is not None:
        args.ids = None
        groups = args.groups[0]
    else:
        groups = None

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

    # =================== CRPS COMPUTATION =================== #
    # We'll compute the CRPS for each run, on each fold.
    # We'll store it in a DataFrame with the following columns:
    # run_id, group, fold, crps
    # Lists to store the columns of the DataFrame
    col_run_id, col_group, col_fold, col_crps = [], [], [], []
    for run_id in run_ids:
        # Compute the CRPS for the run using the 'CRPS' metric function
        # of the run's PredictionDistribution object
        distrib = all_tasks[run_id]["vmax"]["distrib_obj"]
        predictions = all_predictions[run_id]["vmax"]
        targets = all_targets[run_id]["vmax"]
        crps = distrib.metrics["CRPS"](predictions, targets, reduce_mean="all")
        # Store the results in the future columns of the DataFrame
        col_run_id.append(run_id)
        col_group.append(runs[run_id].group)
        col_fold.append(configs[run_id]["experiment"]["fold"])
        col_crps.append(crps)
    # Assemble the DataFrame
    results = pd.DataFrame(
        {
            "run_id": col_run_id,
            "group": col_group,
            "fold": col_fold,
            "crps": col_crps,
        }
    )
    # Compute the mean and the std of the CRPS for each group
    # to obtain a DF (group, mean_crps, std_crps)
    group_results = results.groupby("group").agg({"crps": ["mean", "std"]})
    group_results.columns = group_results.columns.droplevel()
    group_results.columns = ["mean_crps", "std_crps"]
    group_results = group_results.reset_index()
    # Save the results to a CSV file
    results.to_csv(results_dir / "results.csv", index=False)
