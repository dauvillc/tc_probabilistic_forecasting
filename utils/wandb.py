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
    configs: Mapping run_name -> config
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
    run_configs = {}  # Mapping run_name -> config
    run_tasks = {}  # Mapping run_name -> tasks
    for run, run_id in zip(runs, run_ids):
        # Retrieve the config from the run
        cfg = run.config
        run_configs[run_id] = cfg
        # ===== TASKS DEFINITION ==== #
        # Create the tasks
        tasks = create_tasks(cfg)
        run_tasks[run_id] = tasks

        # ===== DATA LOADING ===== #
        # The dataset contains the same samples for every experiment, but not necessarily
        # the same tasks (although some must be in common).
        # That means we have to recreate the dataset for each experiment.
        train_dataset, val_dataset, _, val_loader = load_dataset(cfg, input_variables, tasks, ['tcir'])

        # ===== MODEL RECONSTUCTION ===== #
        # Retrieve the checkpoint from wandb
        artifact = current_run.use_artifact(f'arches/tc_prediction/model-{run_id}:best')
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
                                                          cfg=cfg)
        trainer = pl.Trainer(accelerator='gpu')

        # ===== MAKING PREDICTIONS ===== #
        # Compute the predictions on the validation set
        model_predictions = trainer.predict(model, val_loader)
        # Right now, the predictions are stored as a list of batches. Each batch
        # is a dictionary mapping task -> predictions.
        concatenated_predictions = {}
        for task in model_predictions[0].keys():
            # The predictions can be either a single tensor, or a tuple of tensors
            # (for the multivariate normal distribution)
            if isinstance(model_predictions[0][task], tuple):
                n_tensors = len(model_predictions[0][task])
                concatenated_predictions[task] = tuple(torch.cat([batch[task][i] for batch in model_predictions])
                                                         for i in range(n_tensors))
            else:
                concatenated_predictions[task] = torch.cat([batch[task] for batch in model_predictions])

        # Store the predictions of that model
        predictions[run_id] = concatenated_predictions

    # We also need to save the targets for each task
    # If the targets for a task are already stored, we don't need to do anything
    for task in predictions[run_id].keys():
        if task not in targets.keys():
            targets[task] = torch.cat([targets_batch[task].cpu() for _, _, targets_batch, _ in val_loader])
    # The dataset yields normalized targets, so we need to denormalize them to compute the metrics
    # Remark: the normalization constants were computed on the training set.
    targets = val_dataset.denormalize_tabular_target(targets)

    return run_configs, run_tasks, predictions, targets

