"""
Retrieves a model from a W&B run and makes predictions on the validation or test set.
"""

import sys

sys.path.append("./")
import os
import pytorch_lightning as pl
import argparse
import wandb
from pathlib import Path
from models.lightning_structure import StormPredictionModel
from data_processing.assemble_experiment_dataset import load_dataset
from utils.predictions import ResidualPrediction
from utils.wandb import retrieve_wandb_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions on the validation or test set.")
    parser.add_argument(
        "-i", "--run_id", type=str, default=None, help="The run id of the model to use."
    )
    parser.add_argument(
        "-g",
        "--group",
        type=str,
        default=None,
        help="Group of W&B runs for which to perform predictions.\
                Must not be used with --run_id.",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=4, help="Number of workers to use for data loading."
    )
    args = parser.parse_args()
    run_ids = [args.run_id] if args.run_id is not None else None
    group = args.group
    if run_ids is None and group is None:
        raise ValueError("Either --run_id or --group must be provided.")
    if run_ids is not None and group is not None:
        raise ValueError("--run_id and --group cannot be used together.")

    # Retrieve the runs config from W&B
    runs, cfgs, all_tasks = retrieve_wandb_runs(run_ids, group)
    # Update the list of run_ids in case "group" was used
    run_ids = list(runs.keys())
    print(f"Found {len(run_ids)} runs.")
    for run_id in run_ids:
        print("Treating run ", run_id)
        run, cfg, run_tasks = runs[run_id], cfgs[run_id], all_tasks[run_id]

        # Create a folder to store the predictions
        save_path = Path("data/predictions") / run_id
        save_dir = save_path / "predictions"
        os.makedirs(save_dir, exist_ok=True)
        # Create a folder to store the targets
        targets_dir = save_path / "targets"
        os.makedirs(targets_dir, exist_ok=True)
        # Set the number of workers for data loading
        cfg["training_settings"]["num_workers"] = args.workers

        # Initialize W&B
        current_run = wandb.init(project="tc_prediction", name=f"pred-{run.name}", job_type="pred")

        # ===== DATA LOADING ===== #
        input_variables = cfg["input_variables"]
        val_dataset, val_loader = load_dataset(
            cfg, input_variables, run_tasks, "val", fold=cfg["experiment"]["fold"]
        )

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
            tabular_tasks=run_tasks,
            dataset=val_dataset,
            cfg=cfg,
        )
        trainer = pl.Trainer(accelerator="gpu")

        # ===== MAKING PREDICTIONS ===== #
        # Compute the predictions on the validation set
        model_predictions = trainer.predict(model, val_loader)
        # Right now, the predictions are stored as a list of batches. Each batch
        # is a ResidualPrediction object containing the predictions for each task.
        model_predictions = ResidualPrediction.cat(model_predictions)
        # Save the predictions
        model_predictions.save(save_dir)

        # Save the targets - we can actually store them as a ResidualPrediction object
        # since it has the same structure as the predictions.
        print("Saving targets...")
        targets = []
        for _, _, true_locations, true_residuals in val_loader:
            for task in run_tasks:
                targets.append(ResidualPrediction())
                targets[-1].add(task, true_locations[task], true_residuals[task])
        targets = ResidualPrediction.cat(targets)
        # The dataset yields normalized targets, so we need to denormalize them to compute the metrics
        targets = targets.denormalize(val_dataset)
        targets.save(targets_dir)
