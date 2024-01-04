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
    # Select the corresponding targets and predictions
    targets = next(iter(val_loader))[3]['tcir'][random_indices]
    predictions = predictions[0]['tcir'][random_indices]
    # The targets and predictions are tensors of shape (5, 2, T, H, W) (the first
    # channel is IR, the second is Microwave). We'll only show the last time step:
    targets = targets[:, :, -1]
    predictions = predictions[:, :, -1]
    # Create a figure with four columns: the target IR, the predicted IR, the 
    # target Microwave and the predicted Microwave
    fig, axes = plt.subplots(5, 4, figsize=(20, 20))
    for i, (target, prediction) in enumerate(zip(targets, predictions)):
        axes[i, 0].imshow(target[0].cpu(), cmap='gray')
        axes[i, 0].set_title('Target IR')
        axes[i, 1].imshow(prediction[0].cpu(), cmap='gray')
        axes[i, 1].set_title('Predicted IR')
        axes[i, 2].imshow(target[1].cpu(), cmap='gray')
        axes[i, 2].set_title('Target Microwave')
        axes[i, 3].imshow(prediction[1].cpu(), cmap='gray')
        axes[i, 3].set_title('Predicted Microwave')
    # Log the figure
    wandb.log({'val_visualization': wandb.Image(fig)})
