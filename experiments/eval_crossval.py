"""
Evaluates a set of M models, where each model is evaluated on K folds.
"""

import sys

sys.path.append("./")
import argparse
import wandb
import pandas as pd
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

    # =================== RUNS CONFIGURATION =================== #
    # Initialize W&B
    current_run_name = args.name
    current_run = wandb.init(project="tc_prediction", name=current_run_name, job_type="eval")
    # Retrieve the config of each run
    runs, configs, all_tasks = retrieve_wandb_runs(run_ids=args.ids, group=groups)
    run_ids = list(runs.keys())
    # Retrieve the run name for each run id
    run_names = {run_id: configs[run_id]["experiment"]["name"] for run_id in run_ids}

    # =================== DATA LOADING =================== #
    # Load the predictions and targets for every run
    all_predictions, all_targets = load_predictions_and_targets(run_ids)

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
        predictions = all_predictions[run_id]['vmax']
        targets = all_targets[run_id]['vmax']
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
    print(results)
