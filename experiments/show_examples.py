"""
Trains a model on the TC intensity forecasting task.
"""

import sys

sys.path.append("./")
from utils.utils import update_dict
from distributions.multivariate_normal import MultivariateNormal
from distributions.deterministic import DeterministicDistribution
from distributions.normal import NormalDistribution
from distributions.quantile_composite import QuantileCompositeDistribution
from distributions.categorical import CategoricalDistribution
from data_processing.assemble_experiment_dataset import load_dataset
import matplotlib.pyplot as plt
import argparse
import wandb
import yaml
import pytorch_lightning as pl


def create_output_distrib(task_params, cfg):
    """
    Creates the output distribution object, which implements the loss function,
    metrics, CDF and inverse CDF.

    Parameters
    ----------
    task_params : dict
        The parameters of the task, which include the distribution name
        and parameters.
    cfg: dict
        The configuration dictionary.

    Returns
    -------
    distribution : the distribution object.
    """
    distrib_name = task_params["distribution"]
    if distrib_name in ["quantile_composite", "qc"]:
        distribution = QuantileCompositeDistribution()
    elif distrib_name == "normal":
        distribution = NormalDistribution()
    elif distrib_name == "deterministic":
        # Using a dummy distribution that is deterministic allows to use the same
        # code for deterministic and probabilistic models
        distribution = DeterministicDistribution()
    elif distrib_name == "multivariate_normal":
        distribution = MultivariateNormal(len(cfg["experiment"]["target_steps"]))
    elif distrib_name == "categorical":
        distribution = CategoricalDistribution(task_params["num_classes"])
    else:
        raise ValueError(f"Unknown output distribution {distrib_name}.")
    return distribution


def create_tasks(cfg):
    """
    Creates the dictionary of tasks from the configuration file.
    """
    tasks_cfg = cfg["tasks"]
    tasks = {}
    if tasks_cfg is None:
        return tasks
    for task, params in tasks_cfg.items():
        tasks[task] = {
            "output_variables": params["output_variables"],
            "distribution": params["distribution"],
            "predict_residuals": params["predict_residuals"],
            "weight": params["weight"],
        }
        # Create the distribution object, which implements the loss function,
        # the metrics, optionally the activation function
        distrib = create_output_distrib(params, cfg)
        tasks[task]["distrib_obj"] = distrib
        # Retrieve the output size, either:
        #  - the number of output variables if more than one (and then the distribution must be deterministic)
        #  - the number of parameters of the distribution if it is probabilistic (and only one output variable)
        if len(tasks[task]["output_variables"]) > 1:
            if params["distribution"] != "deterministic":
                raise ValueError(
                    f"The distribution {params['distribution']} must be deterministic "
                    f"if the task has more than one output variable."
                )
            tasks[task]["output_size"] = len(params["output_variables"])
        else:
            tasks[task]["output_size"] = distrib.n_parameters
    return tasks


if __name__ == "__main__":
    pl.seed_everything(123)

    # Argument parser
    parser = argparse.ArgumentParser()
    # Add the --sweep flag to indicate that the script is run as part of a sweep
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Flag to indicate that the script is run as part of a sweep.",
    )
    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        help="Cross-validation fold to use for training.",
        required=True,
    )
    args = parser.parse_args()

    # Load the training configuration file
    with open("training_cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
        experiment_cfg = cfg["experiment"]
        training_cfg = cfg["training_settings"]
        model_cfg = cfg["model_hyperparameters"]
        group = experiment_cfg["group"] if "group" in experiment_cfg else None
    # Add the fold to the configuration
    experiment_cfg["fold"] = args.fold
    # Retrieve the input variables
    input_variables = experiment_cfg["context_variables"]

    # Load the project configuration file
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Modifications in case the script is run as part of a sweep
    if args.sweep:
        current_run = wandb.init(
            project="tc_prediction", config=cfg, group=group, dir=config["paths"]["wandb_logs"]
        )
        # Initialize W&B
        # Replace the default values of the configuration file by the ones from the sweep
        cfg = update_dict(cfg, wandb.config["sweep_parameters"])
    else:
        current_run = wandb.init(
            project="tc_prediction",
            name=experiment_cfg["name"],
            group=group,
            dir=config["paths"]["wandb_logs"],
        )

    # ====== TASKS DEFINITION ====== #
    tasks = create_tasks(cfg)

    # ====== DATA LOADING ====== #
    train_dataset, train_loader = load_dataset(
        cfg, input_variables, tasks, "train", fold=args.fold
    )
    val_dataset, val_loader = load_dataset(cfg, input_variables, tasks, "val", fold=args.fold)

    # ====== EXAMPLES VISUALIZATION ====== #
    # Retrieve the first batch of the training dataset
    for batch in train_loader:
        break 
    past_variables, past_datacubes, target_locations, target_residuals = batch
    past_datacubes = past_datacubes['tcir']
    # Denormalize the target variables
    true_locations = train_dataset.denormalize_tabular_target(target_locations)['vmax']
    # past_datacubes is a sequence of images, of shape (batch_size, channels, sequence_length, height, width)
    # Create a figure with one row per example and one column per time step
    n_examples = 30
    _, C, T, H, W = past_datacubes.shape
    fig, axs = plt.subplots(n_examples, T, figsize=(T, n_examples))
    for i in range(n_examples):
        for t in range(T):
            axs[i, t].imshow(past_datacubes[i, 0, t], cmap="gray")
            axs[i, t].axis("off")
            # Print the target location as title of each subplot
            axs[i, t].set_title(f"{true_locations[i, t].item():.1f} kts")
    plt.tight_layout()
    plt.savefig("figures/past_datacubes.png")

    # Save every example image in a separate file
    # Do it for both channels
    for i in range(n_examples):
        for t in range(T):
            # Channel 0
            plt.imshow(past_datacubes[i, 0, t], cmap="gray")
            plt.axis("off")
            # Remove all margins
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(f"figures/examples/past_datacubes_{i}_t{t}_c0.png", bbox_inches="tight")
            plt.close()
            # Channel 1
            plt.imshow(past_datacubes[i, 1, t], cmap="gray")
            plt.axis("off")
            # Remove all margins
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(f"figures/examples/past_datacubes_{i}_t{t}_c1.png", bbox_inches="tight")
            plt.close()
