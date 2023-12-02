"""
Uses a CNN with the Multiple Quantile Loss.
"""
import sys
sys.path.append("./")
import argparse
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from data_processing.assemble_experiment_dataset import load_dataset
from models.main_structure import StormPredictionModel
from models.cnn3d import CNN3D
from models.variables_projection import VectorProjection3D
from utils.loss_functions import MultipleQuantileLoss
from utils.metrics import QuantilesCRPS


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
    quantiles = np.arange(0.05, 1, 0.05) 
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True,
                        help="The name of the experiment.")
    parser.add_argument("-p", "--past_steps", type=int, default=4,
                        help="Number of time steps given as input to the model. Must be >= 3.")
    parser.add_argument("-n", "--future_steps", type=int, default=4,
                        help="Number of time steps to predict.")
    parser.add_argument("--hidden_channels", type=int, default=4,
                        help="Number of channels in the first convolutional layer.")
    parser.add_argument("--depth" , type=int, default=5,
                        help="Number of convolutional blocks.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train the model for.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size.")
    parser.add_argument("--input_data", type=str, default="era5+hursat",
                        help="The input data to use. Can be 'era5', 'hursat' or 'era5+hursat'.")
    args = parser.parse_args()
    past_steps, future_steps = args.past_steps, args.future_steps
    epochs = args.epochs
    if past_steps < 3:
        raise ValueError("The number of past steps must be >= 3.")

    # ====== DATA LOADING ====== #
    train_dataset, val_dataset, train_loader, val_loader = load_dataset(args, input_variables, output_variables)

    # ====== W+B LOGGER ====== #
    # Initialize the W+B logger
    wandb_logger = WandbLogger(project="tc_prediction", name=args.name, log_model="all")
    # Log the hyperparameters
    wandb_logger.log_hyperparams(args)
    wandb_logger.log_hyperparams({"input_variables": input_variables,
                                    "output_variables": output_variables,
                                    "quantiles": quantiles})

    # ====== MODELS CREATION ====== #
    # Create the loss function
    loss_function = MultipleQuantileLoss(quantiles=quantiles, reduction="mean")
    # Additional metrics to track:
    metrics = {}
    # We're interested in the higher quantiles, so we'll also track the MQL for all quantiles
    # higher than a certain threshold.
    min_quantiles = [0.5, 0.75, 0.9, 0.95]
    for q in min_quantiles:
        metrics[f"MQL_{q}"] = MultipleQuantileLoss(quantiles, reduction="mean", min_quantile=q)
    # We'll also track the CRPS, and consider the min wind speed to be 0 and the max to be 100 m/s
    metrics["CRPS"] = QuantilesCRPS(quantiles, 0, 100)

    # Initialize the model
    patch_size = train_dataset.patch_size()
    datacube_size = (past_steps,) + patch_size
    channels = train_dataset.datacube_channels()
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

