"""
Retrieves a model from a W&B run and makes predictions on the validation or test set.
"""

import sys

sys.path.append("./")
import os
import torch
import pytorch_lightning as pl
import argparse
import wandb
from pathlib import Path
from models.lightning_structure import StormPredictionModel
from data_processing.assemble_experiment_dataset import load_dataset
from utils.wandb import retrieve_wandb_runs
from utils.io import write_tensors_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions on the validation or test set.")
    parser.add_argument(
        "-i", "--run_id", type=str, required=True, help="The run id of the model to use."
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=4, help="Number of workers to use for data loading."
    )
    parser.add_argument("--save_targets", action="store_true", help="Whether to save the targets.")
    args = parser.parse_args()
    run_id = args.run_id

    # Create a folder to store the predictions
    save_dir = os.path.join("data", "predictions", run_id)
    os.makedirs(save_dir, exist_ok=True)

    # Retrieve the run config from W&B
    runs, cfgs, tasks = retrieve_wandb_runs([run_id])
    run, cfg, tasks = runs[run_id], cfgs[run_id], tasks[run_id]
    # Set the number of workers for data loading
    cfg["training_settings"]["num_workers"] = args.workers

    # Initialize W&B
    current_run = wandb.init(project="tc_prediction", name=f"pred-{run.name}", job_type="pred")

    # ===== DATA LOADING ===== #
    input_variables = cfg["input_variables"]
    train_dataset, val_dataset, _, val_loader = load_dataset(cfg, input_variables, tasks, ["tcir"])

    # ===== MODEL RECONSTUCTION ===== #
    # Retrieve the checkpoint from wandb
    artifact = current_run.use_artifact(f"arches/tc_prediction/model-{run_id}:best")
    artifact_dir = artifact.download("/home/cdauvill/scratch/artifacts/")
    checkpoint = Path(artifact_dir) / "model.ckpt"
    # Reconstruct the model from the checkpoint
    datacube_shape = val_dataset.datacube_shape("tcir")
    num_input_variables = len(input_variables)
    model = StormPredictionModel.load_from_checkpoint(
        checkpoint,
        input_datacube_shape=datacube_shape,
        num_input_variables=num_input_variables,
        tabular_tasks=tasks,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=cfg,
    )
    trainer = pl.Trainer(accelerator="gpu")

    # ===== MAKING PREDICTIONS ===== #
    # Compute the predictions on the validation set
    model_predictions = trainer.predict(model, val_loader)
    # Right now, the predictions are stored as a list of batches. Each batch
    # is a dictionary mapping task -> predictions.
    predictions = {}
    for task in model_predictions[0].keys():  # [0] to get the first batch
        # The predictions can be either a single tensor, or a tuple of tensors
        # (for the multivariate normal distribution)
        if isinstance(model_predictions[0][task], tuple):
            n_tensors = len(model_predictions[0][task])
            predictions[task] = tuple(
                torch.cat([batch[task][i] for batch in model_predictions])
                for i in range(n_tensors)
            )
        else:
            predictions[task] = torch.cat([batch[task] for batch in model_predictions])

    # Save the predictions
    write_tensors_dict(predictions, save_dir)

    # Save the targets if requested
    if args.save_targets:
        print("Saving targets...")
        # Create a folder to store the targets
        targets_dir = os.path.join("data", "targets")
        targets = {}
        for task in model_predictions[0].keys():
            targets[task] = torch.cat([batch_target[task] for _, _, batch_target, _ in val_loader])

        # The dataset yields normalized targets, so we need to denormalize them to compute the metrics
        # Remark: the normalization constants were computed on the training set.
        targets = val_dataset.denormalize_tabular_target(targets)
        write_tensors_dict(
            targets,
            targets_dir,
        )
