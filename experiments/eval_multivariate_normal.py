"""
Evaluates a set of probabilistic models from W&B runs. 
"""
import sys
sys.path.append("./")
import argparse
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.wandb import make_predictions


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ids", required=True, nargs='+',
                        help="The ids of the experiment to evaluate.")
    args = parser.parse_args()

    # Initialize W&B 
    current_run = wandb.init(project="tc_prediction",
                             name="eval-" + "-".join(args.ids),
                             job_type="eval")

    # Make predictions using the models from the runs
    configs, predictions, targets = make_predictions(args.ids, current_run)
    # Retrieve the run name for each run id
    run_names = [configs[run_id]['experiment']['name'] for run_id in args.ids]

    # ================= EVALUATION ================= #
    # Not all runs necessarily have the same tasks, so we first need to retrieve
    # the list of all tasks present in the runs.
    tasks = set()
    for run_id in predictions.keys():
        tasks.update(predictions[run_id].keys())
    tasks = list(tasks)

    # Define a grid of values for the intensity
    vmax_points = torch.linspace(0, 200, 50) # In knots
    vmax_grid = torch.meshgrid(vmax_points, vmax_points, indexing='ij')

    # The following plots will be made separately for each model
    for run_id, run_name in zip(args.ids, run_names):
        # First: display a few examples of predicted distributions
        # To do so, we need to take a choose a few random examples from the test set
        sample_idx = np.random.choice(len(targets['vmax']), 5)
        # The predictions are tuples (mean, Cholesky factor of covariance)
        sample_means = predictions[run_id]['vmax'][0][sample_idx]
        sample_Ls = predictions[run_id]['vmax'][1][sample_idx]
        sample_targets = targets['vmax'][sample_idx]
        # Create a figure with one row per example and one column per time step except the first
        future_steps = configs[run_id]['experiment']['future_steps']
        fig, axes = plt.subplots(nrows=len(sample_idx), ncols=future_steps-1, figsize=(15, 10))
        for i, (mean, L, target) in enumerate(zip(sample_means, sample_Ls, sample_targets)):
            # Compute the covariance matrix from the Cholesky factor
            cov = torch.matmul(L, L.transpose(0, 1))
            for t in range(1, future_steps):
                # We'll plot the bivariate distribution P(vmax_1, vmax_t) for each t
                # To do so, we need to select the submatrix of cov corresponding to
                # time steps 1 and t
                cov_1t = cov[[0, 0, t, t], [0, t, 0, t]].reshape(2, 2)
                mean_1t = mean[[0, t]]
                # Compute the pdf on the grid
                points = torch.stack([vmax_grid[0].flatten(), vmax_grid[1].flatten()], dim=1)
                pdf = torch.distributions.MultivariateNormal(mean_1t, cov_1t).log_prob(points)
                pdf = torch.exp(pdf).reshape(vmax_grid[0].shape)
                # Plot the pdf
                axes[i, t-1].contourf(vmax_grid[0], vmax_grid[1], pdf, cmap='Blues')
                # Plot the target (y_1, y_t) as a red cross
                axes[i, t-1].plot(target[0], target[t-1], 'rx')
        # Activate the labels on the last row and first column only
        for i in range(len(sample_idx)):
            axes[i, 0].set_ylabel("VMax at t+6h")
        for t in range(1, future_steps):
            axes[-1, t-1].set_xlabel(f"VMax at t+{6*(t+1)}h")
        # Disable the left ticklabels on all columns except the first
        for i in range(len(sample_idx)):
            for t in range(1, future_steps-1):
                axes[i, t].yaxis.set_tick_params(labelleft=False)
        # Disable the bottom ticklabels on all rows except the last
        for t in range(future_steps-1):
            for i in range(len(sample_idx)-1):
                axes[i, t].xaxis.set_tick_params(labelbottom=False)
        fig.supylabel("Examples")
        fig.suptitle("Examples of predicted distributions $P(VMax_{t+1}, VMax_{t+\\tau})$. Wind speeds are in knots.")

        # Log the figure to W&B
        current_run.log({f"vmax_pdf_examples_{run_name}": wandb.Image(fig)})

