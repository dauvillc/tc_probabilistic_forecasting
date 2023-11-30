"""
Uses a CNN with the Multiple Quantile Loss.
"""
import sys
sys.path.append("./")
import argparse
import torch
import yaml
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from tasks.intensity import intensity_dataset
from data_processing.formats import SuccessiveStepsDataset, datacube_to_tensor
from data_processing.datasets import load_hursat_b1, load_era5_patches
from utils.train_test_split import train_val_test_split
from models.main_structure import StormPredictionModel
from models.cnn3d import CNN3D
from models.variables_projection import VectorProjection3D
from utils.utils import hours_to_sincos
from utils.loss_functions import MultipleQuantileLoss


def create_model(datacube_size, datacube_channels, num_input_variables,
                 predicted_time_steps, n_quantiles,
                 hidden_channels=4, loss_function=None,
                 metrics=None):
    """
    Creates a model for the storm prediction task.

    Parameters
    ----------
    datacube_size : tuple of ints
        The size of the datacube to predict, under the form (D, H, W).
    datacube_channels : int
        The number of channels in the datacube.
    num_input_variables : int
        The number of scalar variables the model receives as input.
    predicted_time_steps : int
        The number of time steps to predict.
    n_quantiles : int
        The number of quantiles to predict.
    hidden_channels : int, optional
        The number of channels in the first convolutional layer.
    loss_function : callable, optional
        The loss function to use. If None, the mean squared error is used.
    metrics: Mapping of str to callable, optional
        The metrics to track. The keys are the names of the metrics, and the values
        are functions that take as input the output of the model and the target,
        and return a scalar.
    """
    # Prediction network (3d CNN + Prediction head)
    cnn_model = CNN3D(datacube_size,
                      input_channels=datacube_channels,
                      output_shape=(predicted_time_steps, n_quantiles),
                      hidden_channels=hidden_channels)
    # Projection network (vector projection + 3d CNN)
    projection_model = VectorProjection3D(num_input_variables,
                                          (datacube_channels, ) + datacube_size)
    # Assemble the main structure, built with Lightning
    model = StormPredictionModel(cnn_model, projection_model, loss_function=loss_function,
                                 metrics=metrics)
    return model


if __name__ == "__main__":
    # Some parameters
    input_variables = ['LAT', 'LON', 'HOUR_SIN', 'HOUR_COS']
    output_variables = ['INTENSITY']
    quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True,
                        help="The name of the experiment.")
    parser.add_argument("-p", "--past_steps", type=int, default=4,
                        help="Number of time steps given as input to the model. Must be >= 3.")
    parser.add_argument("-n", "--prediction_steps", type=int, default=4,
                        help="Number of time steps to predict.")
    parser.add_argument("--hidden_channels", type=int, default=4,
                        help="Number of channels in the first convolutional layer.")
    parser.add_argument("--depth" , type=int, default=5,
                        help="Number of convolutional blocks.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train the model for.")
    parser.add_argument("--input_data", type=str, default="era5+hursat",
                        help="The input data to use. Can be 'era5', 'hursat' or 'era5+hursat'.")
    args = parser.parse_args()
    past_steps, future_steps = args.past_steps, args.prediction_steps
    epochs = args.epochs
    if past_steps < 3:
        raise ValueError("The number of past steps must be >= 3.")
    # Load the configuration file
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    # ====== W+B LOGGER ====== #
    # Initialize the W+B logger
    wandb_logger = WandbLogger(project="tc_prediction", name=args.name)
    # Log the hyperparameters
    wandb_logger.log_hyperparams(args)

    # ====== DATA LOADING ====== #
    # Load the trajectory forecasting dataset
    all_trajs = intensity_dataset()
    # Add a column with the sin/cos encoding of the hours, which will be used as input
    # to the model
    sincos_hours = hours_to_sincos(all_trajs['ISO_TIME'])
    all_trajs['HOUR_SIN'], all_trajs['HOUR_COS'] = sincos_hours[:, 0], sincos_hours[:, 1] 

    # Load the HURSAT-B1 data associated to the dataset
    # We need to load the hursat data even if we don't use it, because we need to
    # keep only the storms for which we have HURSAT-B1 data to fairly compare the
    # runs.
    found_storms, hursat_data = load_hursat_b1(all_trajs, use_cache=True, verbose=True)
    # Keep only the storms for which we have HURSAT-B1 data
    all_trajs = all_trajs.merge(found_storms, on=['SID', 'ISO_TIME'])
    # Load the right patches depending on the input data
    if args.input_data == "era5":
        # Load the ERA5 patches associated to the dataset
        atmo_patches, surface_patches = load_era5_patches(all_trajs, load_atmo=False)
        full_patches = datacube_to_tensor(surface_patches)
    elif args.input_data == "hursat":
        full_patches = datacube_to_tensor(hursat_data)
    elif args.input_data == "era5+hursat":
        # Load the ERA5 patches associated to the dataset
        atmo_patches, surface_patches = load_era5_patches(all_trajs, load_atmo=False)
        era5_patches = datacube_to_tensor(surface_patches)
        hursat_patches = datacube_to_tensor(hursat_data)
        # Concatenate the patches along the channel dimension
        full_patches = torch.cat([era5_patches, hursat_patches], dim=1)
    else:
        raise ValueError("The input data must be 'era5', 'hursat' or 'era5+hursat'.")


    # ====== TRAIN/VAL/TEST SPLIT ====== #
    # Split the dataset into train, validation and test sets
    train_index, val_index, test_index = train_val_test_split(all_trajs,
                                                              train_size=0.6,
                                                              val_size=0.2,
                                                              test_size=0.2)
    train_trajs = all_trajs.iloc[train_index]
    val_trajs = all_trajs.iloc[val_index]
    test_trajs = all_trajs.iloc[test_index]

    print(f"Number of trajectories in the training set: {len(train_trajs)}")
    print(f"Number of trajectories in the validation set: {len(val_trajs)}")
    print(f"Number of trajectories in the test set: {len(test_trajs)}")

    # ====== DATASET CREATION ====== #

    def create_dataloaders(patches, batch_size=128):
        """
        Creates the train and validation dataloaders for the given patches.
        """
        # Split the patches into train, validation and test sets
        train_patches = patches[train_index]
        val_patches = patches[val_index]

        # Create the train and validation datasets. For the validation dataset,
        # we need to normalise the data using the statistics from the train dataset.
        yield_input_variables = len(input_variables) > 0
        train_dataset = SuccessiveStepsDataset(train_trajs, train_patches, past_steps, future_steps,
                                               input_variables, output_variables,
                                               yield_input_variables=yield_input_variables)
        val_dataset = SuccessiveStepsDataset(val_trajs, val_patches, past_steps, future_steps,
                                             input_variables, output_variables,
                                             yield_input_variables=yield_input_variables,
                                             normalise_from=train_dataset)
        # Create the train and validation data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    # Instantiate the train and validation data loader
    batch_size = 128
    train_loader, val_loader = create_dataloaders(full_patches, batch_size=batch_size)

    # ====== MODELS CREATION ====== #
    # Create the loss function
    all_intensities = all_trajs['INTENSITY'].values
    loss_function = MultipleQuantileLoss(quantiles=quantiles, reduction="mean")
    # Additional metrics to track:
    metrics = {}
    # We're interested in the higher quantiles, so we'll also track the MQL for all quantiles
    # higher than a certain threshold.
    min_quantiles = [0.5, 0.75, 0.9, 0.95]
    for q in min_quantiles:
        metrics[f"MQL_{q}"] = MultipleQuantileLoss(quantiles, reduction="mean", min_quantile=q)

    # Initialize the model
    patch_size = full_patches.shape[-2:]
    datacube_size = (past_steps,) + patch_size
    channels = full_patches.shape[1]
    # The number of scalar variables the model receives is the number of variables
    # (e.g. 2 for lat/lon) times the number of past steps
    num_input_variables = len(input_variables) * past_steps
    model =  create_model(datacube_size, channels, num_input_variables,
                          future_steps, len(quantiles),
                          loss_function=loss_function,
                          hidden_channels=args.hidden_channels,
                          metrics=metrics)

    # ====== MODELS TRAINING ====== #
    # Train the models. Save the train and validation losses
    trainer = pl.Trainer(accelerator='gpu', precision="bf16-mixed",
                         max_epochs=epochs, logger=wandb_logger,
                         callbacks=[ModelCheckpoint(monitor='val_loss', mode='min')])
    trainer.fit(model, train_loader, val_loader)

