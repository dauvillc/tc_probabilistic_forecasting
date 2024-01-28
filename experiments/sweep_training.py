"""
Trains a model on a set of hyperparameters defined by a W&B sweep.
"""
import sys
sys.path.append("./")
import pytorch_lightning as pl
import yaml
import wandb
import collections
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from models.lightning_structure import StormPredictionModel
from data_processing.assemble_experiment_dataset import load_dataset
from distributions.quantile_composite import QuantileCompositeDistribution
from distributions.normal import NormalDistribution
from distributions.deterministic import DeterministicDistribution


def create_output_distrib(distrib_name, tasks):
    """
    Creates the output distribution object, which implements the loss function,
    metrics, CDF and inverse CDF.

    Parameters
    ----------
    distrib_name : str
        The name of the distribution to use.
    tasks: dict
        Pointer to the tasks dictionary, which contains the normalization constants.

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
    else:
        raise ValueError(f"Unknown output distribution {distrib_name}.")
    return distribution


def create_tasks(tasks_cfg):
    """
    Creates the dictionary of tasks from the configuration file.
    """
    tasks = {}
    if tasks_cfg is None:
        return tasks
    for task, params in tasks_cfg.items():
        tasks[task] = {'output_variables': params['output_variables'],
                       'distribution': params['distribution']}
        # Create the distribution object, which implements the loss function,
        # the metrics, optionally the activation function
        distrib = create_output_distrib(params['distribution'], tasks)
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


def update(d, u):
    """
    Updates a dictionary recursively.
    Taken from https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def main():
    pl.seed_everything(42)
    # Some parameters
    input_variables = ['LAT', 'LON', 'HOUR_SIN', 'HOUR_COS']

    # Load the training configuration file
    with open("training_cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
        experiment_cfg = cfg["experiment"]
        tasks_cfg = cfg["tasks"]
        training_cfg = cfg["training_settings"]
    past_steps = experiment_cfg["past_steps"]

    if past_steps < 3:
        raise ValueError("The number of past steps must be >= 3.")
    
    # Replace the default values of the configuration file by the ones from the sweep
    cfg = update(cfg, wandb.config)
    # Disable data augmentation for the sweep, to make the runs quicker
    cfg['training_settings']['data_augmentation'] = False

    # ====== TASKS DEFINITION ====== #
    tasks = create_tasks(tasks_cfg)

    # ====== DATA LOADING ====== #
    train_dataset, val_dataset, train_loader, val_loader = load_dataset(cfg, input_variables, tasks, ['tcir'])

    # ====== W+B LOGGER ====== #
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
    model = StormPredictionModel(datacube_shape, num_input_variables, tasks,
                                 datacube_task,
                                 train_dataset, val_dataset,
                                 cfg)

    # ====== MODELS TRAINING ====== #
    # Train the models. Save the train and validation losses
    trainer = pl.Trainer(accelerator='gpu', precision=training_cfg['precision'],
                         max_epochs=training_cfg['epochs'], logger=wandb_logger,
                         callbacks=[LearningRateMonitor()])
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    # Initialize W+B
    wandb.login()
    wandb.init(project="tc_prediction")
    main()
