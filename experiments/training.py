"""
Uses a CNN with the Multiple Quantile Loss.
"""
import sys
sys.path.append("./")
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
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
        distribution = QuantileCompositeDistribution(0, 90, tasks)
    elif distrib_name == 'normal':
        distribution = NormalDistribution(tasks)
    elif distrib_name == 'deterministic':
        # Using a dummy distribution that is deterministic allows to use the same
        # code for deterministic and probabilistic models
        distribution = DeterministicDistribution(tasks)
    else:
        raise ValueError(f"Unknown output distribution {distrib_name}.")
    return distribution


def create_tasks(tasks_cfg):
    """
    Creates the dictionary of tasks from the configuration file.
    """
    tasks = {}
    for task, params in tasks_cfg.items():
        tasks[task] = {'output_variables': params['output_variables'],
                       'distribution': params['distribution']}
        # Create the distribution object and retrieve:
        distrib = create_output_distrib(params['distribution'], tasks)
        # - The loss function
        tasks[task]['loss_function'] = distrib.loss_function
        # - The denormalization function
        tasks[task]['denormalize'] = distrib.denormalize
        # - The output size, either:
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

    # Load the configuration file
    with open("training_cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
        experiment_cfg = cfg["experiment"]
        tasks_cfg = cfg["tasks"]
        training_cfg = cfg["training_settings"]
        model_cfg = cfg["model_hyperparameters"]
    past_steps, future_steps = experiment_cfg["past_steps"], experiment_cfg["future_steps"]

    if past_steps < 3:
        raise ValueError("The number of past steps must be >= 3.")

    # ====== TASKS DEFINITION ====== #
    tasks = create_tasks(tasks_cfg)

    # ====== DATA LOADING ====== #
    train_dataset, val_dataset, train_loader, val_loader = load_dataset(cfg, input_variables, tasks)

    # ====== W+B LOGGER ====== #
    # Initialize the W+B logger
    wandb_logger = WandbLogger(project="tc_prediction", name=experiment_cfg['name'], log_model="all")
    # Log the config and hyperparameters
    wandb_logger.log_hyperparams(cfg)
    wandb_logger.log_hyperparams({"input_variables": input_variables})
    # Log the normalization constants
    input_means, input_stds, tasks_stats = train_dataset.get_normalization_constants()
    wandb_logger.log_hyperparams({"input_means": input_means})
    wandb_logger.log_hyperparams({"input_stds": input_stds})
    wandb_logger.log_hyperparams(tasks_stats)


    # ====== MODELS CREATION ====== #
    # Initialize the model
    datacube_shape = train_dataset.input_datacube_shape(experiment_cfg['input_data'])
    num_input_variables = len(input_variables)
    model = StormPredictionModel(datacube_shape, num_input_variables, future_steps, tasks)

    # ====== MODELS TRAINING ====== #
    # Train the models. Save the train and validation losses
    trainer = pl.Trainer(accelerator='gpu', precision=training_cfg['precision'],
                         max_epochs=training_cfg['epochs'], logger=wandb_logger,
                         callbacks=[ModelCheckpoint(monitor='val_loss', mode='min')])
    trainer.fit(model, train_loader, val_loader)

