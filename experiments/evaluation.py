"""
Evaluates a model taken from a W&B run. All plots are uploaded to W&B and
saved locally.
"""
import sys
sys.path.append("./")
import argparse
import wandb
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pathlib import Path
from models.lightning_structure import StormPredictionModel
from data_processing.assemble_experiment_dataset import load_dataset
from experiments.training import create_tasks


if __name__ == "__main__":
    pl.seed_everything(42)
    # Some parameters
    input_variables = ['LAT', 'LON', 'HOUR_SIN', 'HOUR_COS']
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True,
                        help="The names of the experiment to evaluate.")
    args = parser.parse_args()

    # ====== WANDB INITIALIZATION ====== #
    # Initialize the W+B logger and retrieve the run that led to the model being
    # evaluated
    run = wandb.init(project="tc_prediction", name="eval_" + args.name, job_type="eval")
    api = wandb.Api()
    # Search for all runs with the same name
    runs = api.runs("arches/tc_prediction", filters={"config.experiment.name": args.name})
    if len(runs) == 0:
        raise ValueError(f"No runs with name {args.name} were found.")
    # If several runs have the same name, we'll use any but print a warning
    if len(runs) > 1:
        print(f"WARNING: Several runs with name {args.name} were found. Using the first one.")
    # Get the run id
    evaluated_run = runs[0]
    run_id = evaluated_run.id
    # We can now retrieve the config of the run
    cfg = evaluated_run.config
    # Log the config
    wandb.log(cfg)

    # ===== TASKS DEFINITION ==== #
    # Retrieve the tasks configuration from the config
    tasks_cfg = cfg['tasks']
    # Create the tasks
    tasks = create_tasks(tasks_cfg)

    # ===== DATA LOADING ===== #
    train_dataset, val_dataset, train_loader, val_loader = load_dataset(cfg, input_variables, tasks, ['tcir'])
   
    # ===== MODEL RECONSTUCTION ===== #
    # Retrieve the checkpoint from wandb
    artifact = run.use_artifact(f'arches/tc_prediction/model-{run_id}:best', type="model")
    artifact_dir = artifact.download()
    checkpoint = Path(artifact_dir) / 'model.ckpt'
    # Reconstruct the model from the checkpoint
    datacube_shape = train_dataset.datacube_shape('tcir')
    num_input_variables = len(input_variables)
    model = StormPredictionModel.load_from_checkpoint(checkpoint,
                                                      input_datacube_shape=datacube_shape,
                                                      num_input_variables=num_input_variables,
                                                      tabular_tasks=tasks,
                                                      datacube_tasks={'tcir': datacube_shape},
                                                      cfg=cfg)
    trainer = pl.Trainer(accelerator='gpu')

    # ===== VISUALIZATION ===== #
    # Compute the predictions on the validation set
    predictions = trainer.predict(model, val_loader)
    # Select five random samples from the first batch
    random_indices = torch.randint(0, cfg['training_settings']['batch_size'], (5,))
    # Select the corresponding input, targets and predictions
    batch = next(iter(val_loader))
    inputs = batch[1]['tcir'][random_indices]
    targets = batch[3]['tcir'][random_indices]
    predictions = predictions[0]['tcir'][random_indices]
    # Retrieve the time steps
    past_steps = cfg['experiment']['past_steps']
    future_steps = cfg['experiment']['future_steps']
    # Create a figure with two rows per sample
    # The first row contains input 1, ..., input past_steps, target 1, prediction 1
    # for the IR channel
    # The second row contains input 1, ..., input past_steps, target 1, prediction 1
    # for the PMW channel
    fig, axes = plt.subplots(10, past_steps + 2, figsize=(20, 20))
    # For each sample
    for i in range(0, 10, 2):
        # Plot the IR inputs
        for j in range(past_steps):
            axes[i, j].imshow(inputs[i // 2, 0, j].cpu())
            axes[i, j].set_title(f"IR input {j + 1}")
        # Plot the PMW inputs
        for j in range(past_steps):
            axes[i + 1, j].imshow(inputs[i // 2, 1, j].cpu())
            axes[i + 1, j].set_title(f"PMW input {j + 1}")
        # Plot the IR target
        axes[i, past_steps].imshow(targets[i // 2, 0, 0].cpu())
        axes[i, past_steps].set_title("IR target")
        # Plot the PMW target
        axes[i + 1, past_steps].imshow(targets[i // 2, 1, 0].cpu())
        axes[i + 1, past_steps].set_title("PMW target")
        # Plot the IR prediction
        axes[i, past_steps + 1].imshow(predictions[i // 2, 0, 0].cpu())
        axes[i, past_steps + 1].set_title("IR prediction")
        # Plot the PMW prediction
        axes[i + 1, past_steps + 1].imshow(predictions[i // 2, 1, 0].cpu())
        axes[i + 1, past_steps + 1].set_title("PMW prediction")
    # Save the figure
    wandb.log({"predictions": wandb.Image(fig)})


