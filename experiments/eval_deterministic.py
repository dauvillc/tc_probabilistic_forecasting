"""
Evaluates a model taken from a W&B run. All plots are uploaded to W&B and
saved locally.
"""
import sys
sys.path.append("./")
import argparse
import wandb
from utils.wandb import make_predictions
from plotting.deterministic import plot_deterministic_metrics


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
    predictions, targets = make_predictions(args.ids, current_run)

    # ================= EVALUATION ================= #
    # Not all runs necessarily have the same tasks, so we first need to retrieve
    # the list of all tasks present in the runs.
    tasks = set()
    for run_name in predictions.keys():
        tasks.update(predictions[run_name].keys())
    tasks = list(tasks)
    
    # Plot the RMSE and MAE for each task
    # For the intensity
    fig = plot_deterministic_metrics({name: pred['vmax'] for name, pred in predictions.items()},
                                     targets['vmax'], "Forecast error for the 1-min maximum wind speed",
                                     "knots", save_path=f"figures/evaluation/{current_run.name}/vmax.png")
    current_run.log({"vmax": wandb.Image(fig)})
