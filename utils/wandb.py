"""
Implements some functions to retrieve data from wandb.
"""
import wandb
import pytorch_lightning as pl
import torch
from pathlib import Path
from experiments.training import create_tasks
from models.lightning_structure import StormPredictionModel
from data_processing.assemble_experiment_dataset import load_dataset


def make_predictions(run_ids, current_run):
    """
    For a set of W&B run ids, retrieve the corresponding models and make
    predictions on the validation set.
    
    Parameters
    ----------
    run_ids : list of str
        The ids of the runs to retrieve.
    current_run : wandb.Run
        The current run, used to retrieve the artifacts.

    Returns
    -------
    predictions : Mapping run_name -> predictions on the validation set.
        The keys are the names of the runs, and the values are mappings
        task -> predictions. The predictions are returned as the list of all batches
        of predictions on the validation set (not concatenated nor denormalized,
        stored on cpu).
    targets : Mapping task -> targets on the validation set.
    """
    input_variables = ['LAT', 'LON', 'HOUR_SIN', 'HOUR_COS']
    pl.seed_everything(42)

    # ====== WANDB INITIALIZATION ====== #
    api = wandb.Api()
    # Retrieve each run from the ids, and store their names
    runs, names = [], []
    for run_id in run_ids:
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
    for run, run_id in zip(runs, run_ids):
        # Retrieve the config from the run
        cfg = run.config
        run_name = cfg['experiment']['name']
        # ===== TASKS DEFINITION ==== #
        # Create the tasks
        tasks = create_tasks(cfg)

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
        # Right now, the predictions are stored as a list of batches. Each batch
        # is a dictionary mapping task -> predictions. We want to obtain a
        # dictionary mapping task -> predictions where predictions is a single tensor.
        model_predictions = {task: torch.cat([batch[task].cpu() for batch in model_predictions])
                             for task in tasks.keys()}
        # Apply the activation function (specific to each distribution)
        for task_name, task_params in tasks.items():
            distrib = task_params['distrib_obj']
            model_predictions[task_name] = distrib.activation(model_predictions[task_name])
        # Store the predictions of that model
        predictions[run_name] = model_predictions

    # We also need to save the targets for each task
    # If the targets for a task are already stored, we don't need to do anything
    for task in predictions[run_name].keys():
        if task not in targets.keys():
            targets[task] = torch.cat([targets_batch[task].cpu() for _, _, targets_batch, _ in val_loader])
    # The dataset yields normalized targets, so we need to denormalize them to compute the metrics
    # Remark: the normalization constants were computed on the training set.
    targets = val_dataset.denormalize_tabular_target(targets)
        
    return predictions, targets

