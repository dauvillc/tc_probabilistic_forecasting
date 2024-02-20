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
from utils.wandb import make_predictions
from utils.utils import matplotlib_markers, sshs_category


if __name__ == "__main__":
    # Set sns style
    sns.set_style("whitegrid")

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ids", required=True, nargs='+',
                        help="The ids of the experiment to evaluate.")
    parser.add_argument("-n", "--n_examples", type=int, default=5,
                        help="The number of examples to show for each run.")
    args = parser.parse_args()

    # Initialize W&B
    current_run_name = "examples-" + "-".join(args.ids)
    current_run = wandb.init(project="tc_prediction",
                             name=current_run_name,
                             job_type="eval")

    # Make predictions using the models from the runs
    all_runs_configs, all_runs_tasks, all_runs_predictions, targets = make_predictions(
        args.ids, current_run)
    # Retrieve the run name for each run id
    run_names = {run_id: all_runs_configs[run_id]
                 ['experiment']['name'] for run_id in args.ids}

    # Create a folder to store the plots
    save_folder = f"figures/evaluation/{current_run_name}"
    os.makedirs(save_folder, exist_ok=True)

    # Create a list of markers and colors for the plots. Each run will be represented
    # by a the same marker and color across all plots.
    markers = matplotlib_markers(len(args.ids))
    markers = {run_id: marker for run_id, marker in zip(args.ids, markers)}
    cmap = plt.get_cmap('tab10', len(args.ids))
    colors = {run_id: cmap(k) for k, run_id in enumerate(args.ids)}

    # First, we need to retrieve the list of all tasks that are performed by at least
    # one model.
    all_tasks = set()
    for tasks in all_runs_tasks.values():
        all_tasks.update(list(tasks.keys()))
    all_tasks = list(all_tasks)
    # Number of samples and predicted time steps
    N, T = targets[all_tasks[0]].shape

    # Draw a set of random examples
    idxs = torch.randperm(N)[:args.n_examples]

    # For each task, show the predictions of each run.
    # We'll make one figure for each task.
    # Each figure will have one row for each example and one column for each time step.
    # Each subplot will show the pdf predicted by each run, plus the target.
    for task in all_tasks:
        fig, axs = plt.subplots(nrows=args.n_examples, ncols=T, figsize=(20, 5*args.n_examples))
        fig.suptitle(f"{task}", fontsize=16)
        # Define for which x values we'll plot the pdfs
        # We'll use the min and max of the target values, with some margin
        x_min = targets[task].min().item() * 0.9
        x_max = targets[task].max().item() * 1.1
        x = torch.linspace(x_min, x_max, 100).unsqueeze(1).repeat(1, T) # (100, T)
        # For each example
        for i, idx in enumerate(idxs):
            # Compute the PDF according to each model
            for run_id in args.ids:
                # Retrieve the pdf function from the distribution object
                # associated with the model and the task
                pdf_fn = all_runs_tasks[run_id][task]['distrib_obj'].pdf
                # Plot the predicted pdf
                pdf_vals = pdf_fn(all_runs_predictions[run_id][task][i], x)
                for t in range(T):
                    ax = axs[i, t]
                    ax.plot(x[:, t], pdf_vals[:, t], label=run_names[run_id],
                            marker=markers[run_id], color=colors[run_id],
                            markevery=10)
            # For each time step, plot the target as a vertical line
            for t in range(T):
                axs[i, t].axvline(targets[task][idx, t].item(), color="black", linestyle="--",
                                  label="target")
        # Add a legend to the first subplot
        axs[0, 0].legend()
        # Save the figure
        plt.savefig(f"{save_folder}/{task}.png")
        current_run.log({"examples": wandb.Image(f"{save_folder}/{task}.png")})
        plt.close(fig)

