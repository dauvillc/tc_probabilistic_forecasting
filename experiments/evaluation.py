"""
Evaluates a model taken from a W&B run. All plots are uploaded to W&B and
saved locally.
"""
import sys
sys.path.append("./")
import argparse
import torch
import wandb
import pytorch_lightning as pl
from pathlib import Path
from models.lightning_structure import StormPredictionModel
from data_processing.assemble_experiment_dataset import load_dataset
from experiments.training import create_tasks
from plotting.deterministic import plot_deterministic_metrics


if __name__ == "__main__":
    pl.seed_everything(42)
    # Some parameters
    input_variables = ['LAT', 'LON', 'HOUR_SIN', 'HOUR_COS']
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ids", required=True, nargs='+',
                        help="The ids of the experiment to evaluate.")
    args = parser.parse_args()

    # ====== WANDB INITIALIZATION ====== #
    # Initialize W&B 
    current_run = wandb.init(project="tc_prediction",
                             name="eval-" + "-".join(args.ids),
                             job_type="eval")
    api = wandb.Api()
    # Retrieve each run from the ids, and store their names
    runs, names = [], []
    for run_id in args.ids:
        try:
            runs.append(api.run(f"arches/tc_prediction/{run_id}"))
            names.append(runs[-1].config['experiment']['name'])
        except wandb.errors.CommError:
            print(f"WARNING: Run {run_id} could not be retrieved.")

    # ================= MAKING PREDICTIONS ================= #
    # We'll now make predictions with every model on the same validation set,
    # before comparing them.
    # To do so we need to reconstruct the model from the checkpoint, and then
    # compute the predictions, and store them on cpu.
    predictions = {}  # Mapping run_name -> predictions
    targets = {}  # Mapping task -> targets
    for run, run_id in zip(runs, args.ids):
        # Retrieve the config from the run
        cfg = run.config
        run_name = cfg['experiment']['name']
        # ===== TASKS DEFINITION ==== #
        # Retrieve the tasks configuration from the config
        tasks_cfg = cfg['tasks']
        # Create the tasks
        tasks = create_tasks(tasks_cfg)

        # ===== DATA LOADING ===== #
        # The dataset contains the same samples for every experiment, but not necessarily
        # the same tasks (although some must be in common).
        # That means we have to recreate the dataset for each experiment.
        train_dataset, val_dataset, _, val_loader = load_dataset(cfg, input_variables, tasks, ['tcir'])
       
        # ===== MODEL RECONSTUCTION ===== #
        # Retrieve the checkpoint from wandb
        artifact = current_run.use_artifact(f'arches/tc_prediction/model-{run_id}:latest')
        artifact_dir = artifact.download('/home/cdauvill/scratch/artifacts/')
        checkpoint = Path(artifact_dir) / 'model.ckpt'
        # Reconstruct the model from the checkpoint
        datacube_shape = val_dataset.datacube_shape('tcir')
        num_input_variables = len(input_variables)
        model = StormPredictionModel.load_from_checkpoint(checkpoint,
                                                          input_datacube_shape=datacube_shape,
                                                          num_input_variables=num_input_variables,
                                                          tabular_tasks=tasks,
                                                          train_dataset=train_dataset,
                                                          val_dataset=val_dataset,
                                                          datacube_tasks={},
                                                          cfg=cfg)
        trainer = pl.Trainer(accelerator='gpu')

        # ===== MAKING PREDICTIONS ===== #
        # Compute the predictions on the validation set
        model_predictions = trainer.predict(model, val_loader)
        # Right now the predictions are stored as batches of Mappping[str -> torch.Tensor],
        # but we want to store them as a single Mapping[str -> torch.Tensor], so we need
        # to concatenate the batches.
        # We also need to move the predictions to cpu as not to overload the gpu.
        model_predictions = {task: torch.cat([batch[task].cpu() for batch in model_predictions])
                             for task in tasks}
        predictions[run_name] = model_predictions
        # We also need to save the targets for each task
        # If the targets for a task are already stored, we don't need to do anything
        for task in predictions[run_name].keys():
            if task not in targets.keys():
                targets[task] = torch.cat([targets_batch[task].cpu() for _, _, targets_batch, _ in val_loader])
        # The dataset yields normalized targets, so we need to denormalize them to compute the metrics
        # Remark: the normalization constants were computed on the training set.
        targets = val_dataset.denormalize_tabular_target(targets)

    # ================= EVALUATION ================= #
    # Not all runs necessarily have the same tasks, so we first need to retrieve
    # the list of all tasks present in the runs.
    tasks = set()
    for run in runs:
        tasks.update(predictions[run.config['experiment']['name']].keys())
    tasks = list(tasks)
    
    # Plot the RMSE and MAE for each task
    # For the intensity
    fig = plot_deterministic_metrics({name: pred['vmax'] for name, pred in predictions.items()},
                                     targets['vmax'], "Forecast error for the 1-min maximum wind speed",
                                     "m/s", save_path=f"figures/evaluation/{current_run.name}/vmax.png")
    current_run.log({"vmax": wandb.Image(fig)})
