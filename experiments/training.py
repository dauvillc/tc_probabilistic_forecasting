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
from data_processing.assemble_experiment_dataset import load_dataset
from models.lightning_structure import StormPredictionModel
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pathlib import Path
import argparse
import wandb
import yaml
import pytorch_lightning as pl


def create_output_distrib(distrib_name, tasks, cfg):
    """
    Creates the output distribution object, which implements the loss function,
    metrics, CDF and inverse CDF.

    Parameters
    ----------
    distrib_name : str
        The name of the distribution to use.
    tasks: dict
        Pointer to the tasks dictionary, which contains the normalization constants.
    cfg: dict
        The configuration dictionary.

    Returns
    -------
    distribution : the distribution object.
    """
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
        }
        # Create the distribution object, which implements the loss function,
        # the metrics, optionally the activation function
        distrib = create_output_distrib(params["distribution"], tasks, cfg)
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

    # ====== W+B LOGGER ====== #
    # Initialize the W+B logger
    wandb_logger = WandbLogger(log_model="all")
    # Log the config and hyperparameters
    wandb_logger.log_hyperparams(cfg)
    wandb_logger.log_hyperparams({"input_variables": input_variables})

    # ====== MODELS CREATION ====== #
    # Initialize the model
    num_input_variables = len(input_variables)
    datacube_shape = train_dataset.datacube_shape("tcir")
    # If training from scratch, create a new model
    if experiment_cfg["use-pre-trained-id"] is None:
        model = StormPredictionModel(datacube_shape, tasks, train_dataset, cfg)
    # If fine-tuning, load the model from a previous run
    else:
        # Load the model from a previous run
        run_id = experiment_cfg["use-pre-trained-id"]
        try:
            print("Using model from run ", run_id)
            artifact = current_run.use_artifact(f"arches/tc_prediction/model-{run_id}:latest")
            artifact_dir = artifact.download("/home/cdauvill/scratch/artifacts/")
        except wandb.errors.CommError:
            print(f"Could not find the model {run_id} in the W&B artifacts. ")
            sys.exit(1)
        checkpoint = Path(artifact_dir) / "model.ckpt"
        # Reconstruct the model from the checkpoint
        model = StormPredictionModel.load_from_checkpoint(
            checkpoint,
            input_datacube_shape=datacube_shape,
            tabular_tasks=tasks,
            dataset=train_dataset,
            cfg=cfg,
        )

    # ====== MODELS TRAINING ====== #
    # Train the models. Save the train and validation losses
    trainer = pl.Trainer(
        accelerator="gpu",
        precision=training_cfg["precision"],
        max_epochs=training_cfg["epochs"],
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss", mode="min", dirpath=config["paths"]["checkpoints"]
            ),
            LearningRateMonitor(),
        ],
    )
    trainer.fit(model, train_loader, val_loader)
