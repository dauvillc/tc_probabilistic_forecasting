"""
Evaluates the performances of a set of models by showing random examples.
"""

import sys

sys.path.append("./")
import os
import argparse
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from utils.wandb import retrieve_wandb_runs
from utils.io import load_targets, load_predictions
from utils.utils import matplotlib_markers, sshs_category


if __name__ == "__main__":
    # Set the seed with torch
    torch.manual_seed(0)
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
        "-k",
        "--k_examples",
        type=int,
        default=5,
        help="The number of examples to show for each run.",
    )
    parser.add_argument(
        "--min_cat",
        type=int,
        default=-1,
        help="The minimum SSHS category from which to draw examples.",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        required=True,
        help="Name of the evaluation run.",
    )
    parser.add_argument(
        "-r",
        "--recreate_dataset",
        action="store_true",
        help="Recreate the dataset before evluating each model. Required if the I/O of the\
            dataset differs between the models.",
    )
    args = parser.parse_args()

    # ===== RUNS AND CONFIGS =====
    # Initialize W&B
    current_run_name = args.name
    current_run = wandb.init(project="tc_prediction", name=current_run_name, job_type="eval")
    # Retrieve the config of each run
    runs, configs, all_runs_tasks = retrieve_wandb_runs(args.ids)
    # Retrieve the run name for each run id
    run_names = {run_id: runs[run_id].name for run_id in args.ids}

    # ===== LOAD TARGETS AND PREDICTIONS =====
    all_runs_predictions = load_predictions(args.ids)
    targets = load_targets()

    # ==== PLOTTING PREPARATION ====
    # Create a folder to store the plots
    save_folder = f"figures/evaluation/{current_run_name}"
    os.makedirs(save_folder, exist_ok=True)

    # Create a list of markers and colors for the plots. Each run will be represented
    # by a the same marker and color across all plots.
    markers = matplotlib_markers(len(args.ids))
    markers = {run_id: marker for run_id, marker in zip(args.ids, markers)}
    cmap = plt.get_cmap("tab10", len(args.ids))
    colors = {run_id: cmap(k) for k, run_id in enumerate(args.ids)}

    # ==== PLOTTING ====
    # First, we need to retrieve the list of all tasks that are performed by at least
    # one model.
    common_tasks = set()
    for task_dict in all_runs_tasks.values():
        common_tasks.update(list(task_dict.keys()))
    common_tasks = list(common_tasks)
    # Number of samples and predicted time steps
    N, T = targets[common_tasks[0]].shape

    # Isolate the examples that are at least of the required SSHS category
    sample_intensities = targets["vmax"].max(dim=1).values
    cat_mask = sshs_category(sample_intensities) >= args.min_cat
    cat_idxs = torch.where(cat_mask)[0]
    # Randomly select a subset of examples
    idxs = cat_idxs[torch.randperm(cat_idxs.shape[0])[: args.k_examples]]

    # For each task, show the predictions of each run.
    # We'll make one figure for each task.
    # Each figure will have one row for each example and one column for each time step.
    # Each subplot will show the cdf predicted by each run, plus the target.
    for task in common_tasks:
        fig, axs = plt.subplots(nrows=args.k_examples, ncols=T, figsize=(20, 5 * args.k_examples))
        fig.suptitle(f"{task}", fontsize=16)
        # Define for which x values we'll plot the cdfs
        # We'll use the min and max of the target values, with some margin
        x_min = targets[task].min().item() * 0.9
        x_max = targets[task].max().item() * 1.1
        x = torch.linspace(x_min, x_max, 100).unsqueeze(1).repeat(1, T)  # (100, T)
        # For each run, compute the CRPS. We'll add it in the label of the curves.
        crps = {}
        for run_id in args.ids:
            # Compute the CRPS for the current model
            crps_fn = all_runs_tasks[run_id][task]["distrib_obj"].metrics["CRPS"]
            crps[run_id] = crps_fn(
                all_runs_predictions[run_id][task][idxs],
                targets[task][idxs],
                reduce_mean=False,
            )  # (N,)
        # For each example
        for i, idx in enumerate(idxs):
            # Compute the PDF according to each model
            for run_id in args.ids:
                # Retrieve the cdf function from the distribution object
                # associated with the model and the task
                cdf_fn = all_runs_tasks[run_id][task]["distrib_obj"].cdf
                # Plot the predicted cdf
                cdf_vals = cdf_fn(all_runs_predictions[run_id][task][idx], x)
                for t in range(T):
                    axs[i, t].plot(
                        x[:, t],
                        cdf_vals[:, t],
                        label=run_names[run_id] + f" {crps[run_id][i].item():.2f}",
                        marker=markers[run_id],
                        color=colors[run_id],
                        markevery=10,
                    )
            # For each time step, plot the target as a vertical line
            for t in range(T):
                axs[i, t].axvline(
                    targets[task][idx, t].item(),
                    color="black",
                    linestyle="--",
                    label="target",
                )
        # Add the legend to each subplot, to show the CRPS
        for ax in axs.flatten():
            ax.legend()
        # Save the figure
        plt.savefig(f"{save_folder}/{task}.png")
        current_run.log({"examples": wandb.Image(f"{save_folder}/{task}.png")})
        plt.close(fig)
