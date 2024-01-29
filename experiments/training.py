"""
Trains a model on the TC intensity forecasting task.
"""
import sys
sys.path.append("./")
import pytorch_lightning as pl
import yaml
import wandb
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from models.lightning_structure import StormPredictionModel
from data_processing.assemble_experiment_dataset import load_dataset
from distributions.quantile_composite import QuantileCompositeDistribution
from distributions.normal import NormalDistribution
from distributions.deterministic import DeterministicDistribution
from distributions.multivariate_normal import MultivariateNormal


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
    if distrib_name in ['quantile_composite', 'qc']:
        distribution = QuantileCompositeDistribution(0, 90)
    elif distrib_name == 'normal':
        distribution = NormalDistribution()
    elif distrib_name == 'deterministic':
        # Using a dummy distribution that is deterministic allows to use the same
        # code for deterministic and probabilistic models
        distribution = DeterministicDistribution()
    elif distrib_name == "multivariate_normal":
        distribution = MultivariateNormal(cfg['experiment']['future_steps'])
    else:
        raise ValueError(f"Unknown output distribution {distrib_name}.")
    return distribution


def create_tasks(cfg):
    """
    Creates the dictionary of tasks from the configuration file.
    """
    tasks_cfg = cfg['tasks']
    tasks = {}
    if tasks_cfg is None:
        return tasks
    for task, params in tasks_cfg.items():
        tasks[task] = {'output_variables': params['output_variables'],
                       'distribution': params['distribution']}
        # Create the distribution object, which implements the loss function,
        # the metrics, optionally the activation function
        distrib = create_output_distrib(params['distribution'], tasks, cfg)
        tasks[task]['distrib_obj'] = distrib
        # Retrieve the output size, either:
        #  - the number of output variables if more than one (and then the distribution must be deterministic)
        #  - the number of parameters of the distribution if it is probabilistic (and only one output variable)
        if len(tasks[task]['output_variables']) > 1:
            if params['distribution'] != 'deterministic':
                raise ValueError(f"The distribution {params['distribution']} must be deterministic "
                                 f"if the task has more than one output variable.")
            tasks[task]['output_size'] = len(params['output_variables'])
        else:
            tasks[task]['output_size'] = distrib.n_parameters
    return tasks


if __name__ == "__main__":
    pl.seed_everything(42)
    # Some parameters
    input_variables = ['LAT', 'LON', 'HOUR_SIN', 'HOUR_COS']

    # Load the training configuration file
    with open("training_cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
        experiment_cfg = cfg["experiment"]
        training_cfg = cfg["training_settings"]
        model_cfg = cfg["model_hyperparameters"]
    past_steps, future_steps = experiment_cfg["past_steps"], experiment_cfg["future_steps"]

    # Load the project configuration file
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    if past_steps < 3:
        raise ValueError("The number of past steps must be >= 3.")

    # ====== TASKS DEFINITION ====== #
    tasks = create_tasks(cfg)

    # ====== DATA LOADING ====== #
    train_dataset, val_dataset, train_loader, val_loader = load_dataset(cfg, input_variables, tasks, ['tcir'])

    # ====== W+B LOGGER ====== #
    # Initialize W&B
    current_run = wandb.init(project="tc_prediction",
                             name=experiment_cfg['name'],
                             job_type="training")
    # Initialize the W+B logger
    wandb_logger = WandbLogger(project="tc_prediction", name=experiment_cfg['name'], log_model="all")
    # Log the config and hyperparameters
    wandb_logger.log_hyperparams(cfg)
    wandb_logger.log_hyperparams({"input_variables": input_variables})

    # ====== MODELS CREATION ====== #
    # Initialize the model
    num_input_variables = len(input_variables)
    datacube_shape = train_dataset.datacube_shape('tcir')
    # datacube_task = {'tcir': {'output_channels': 2}}
    datacube_task = {}
    # If training from scratch, create a new model
    if experiment_cfg['use-pre-trained-id'] is None:
        model = StormPredictionModel(datacube_shape, num_input_variables, tasks,
                                     datacube_task,
                                     train_dataset, val_dataset,
                                     cfg)
    # If fine-tuning, load the model from a previous run
    else:
        # Load the model from a previous run
        run_id = experiment_cfg['use-pre-trained-id']
        try:
            print("Using model from run ", run_id)
            artifact = current_run.use_artifact(f'arches/tc_prediction/model-{run_id}:latest')
            artifact_dir = artifact.download('/home/cdauvill/scratch/artifacts/')
        except wandb.errors.CommError:
            print(f"Could not find the model {run_id} in the W&B artifacts. ")
            sys.exit(1)
        checkpoint = Path(artifact_dir) / 'model.ckpt'
        # Reconstruct the model from the checkpoint
        model = StormPredictionModel.load_from_checkpoint(checkpoint,
                                                          input_datacube_shape=datacube_shape,
                                                          num_input_variables=num_input_variables,
                                                          tabular_tasks=tasks,
                                                          train_dataset=train_dataset,
                                                          val_dataset=val_dataset,
                                                          datacube_tasks=datacube_task,
                                                          cfg=cfg)

    # ====== MODELS TRAINING ====== #
    # Train the models. Save the train and validation losses
    trainer = pl.Trainer(accelerator='gpu', precision=training_cfg['precision'],
                         max_epochs=training_cfg['epochs'], logger=wandb_logger,
                         callbacks=[ModelCheckpoint(monitor='val_loss', mode='min',
                                                    dirpath=config['paths']['checkpoints']),
                                    LearningRateMonitor()])
    trainer.fit(model, train_loader, val_loader)

