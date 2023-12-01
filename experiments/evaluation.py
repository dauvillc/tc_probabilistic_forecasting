"""
Evaluates the models.
"""
import sys
sys.path.append("./")
import argparse
import numpy as np
import wandb
import pytorch_lightning as pl
from pathlib import Path
from models.main_structure import StormPredictionModel
from data_processing.assemble_experiment_dataset import load_dataset
from experiments.quantile_regression import create_model
from utils.utils import to_numpy
from utils.metrics import QuantilesCRPS
from plotting.quantiles import plot_quantile_losses


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
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size.")
    parser.add_argument("--input_data", type=str, default="era5+hursat",
                        help="The input data to use. Can be 'era5', 'hursat' or 'era5+hursat'.")
    parser.add_argument('-r', '--ref', type=str, required=True,
                        help="The checkpoint reference to a W&B run.")
    args = parser.parse_args()
    past_steps, future_steps = args.past_steps, args.prediction_steps
    epochs = args.epochs
    if past_steps < 3:
        raise ValueError("The number of past steps must be >= 3.")

    # ====== DATA LOADING ====== #
    train_dataset, val_dataset, train_loader, val_loader = load_dataset(args, input_variables, output_variables)

    # ====== MODEL RECONSTRUCTION ====== #
    # Create the model architecture
    # We need to re-create the architecture, as it is not created within the LightningModule.
    # Once we have recreated it, we can load the weights from the checkpoint.
    patch_size = train_dataset.patch_size()
    datacube_size = (past_steps,) + patch_size
    datacube_channels = train_dataset.datacube_channels()
    num_input_variables = len(input_variables) * past_steps
    model = create_model(datacube_size, datacube_channels, num_input_variables,
                         future_steps, len(quantiles), hidden_channels=args.hidden_channels)
    prediction_model = model.prediction_model
    projection_model = model.projection_model
    
    # Load the model weights
    checkpoint_ref = args.ref
    run = wandb.init(project="tc_prediction", name=args.name, job_type="eval")
    artifact = run.use_artifact(checkpoint_ref, type="model")
    artifact_dir = artifact.download()

    # load checkpoint
    model = StormPredictionModel.load_from_checkpoint(Path(artifact_dir) / "model.ckpt",
                                                      prediction_model=prediction_model,
                                                      projection_model=projection_model)
    # Log the evaluation config
    wandb.log(vars(args))

    # ====== EVALUATION ====== #
    # Make predictions on the validation set
    y_true = val_dataset.future_target
    trainer = pl.Trainer(accelerator="gpu", max_epochs=1)
    pred = np.concatenate([to_numpy(p) for p in trainer.predict(model, val_loader)], axis=0)

    # Compute the CRPS
    crps_computer = QuantilesCRPS(quantiles, 0, 100)
    crps = crps_computer(pred, y_true)
    print("Average CRPS:", crps)
    wandb.log({"avg_crps": crps})

    # Plot the quantile losses and log them to wandb
    fig = plot_quantile_losses({"model": pred}, y_true, quantiles,
                                savepath=f"figures/quantiles/{args.name}_quantile_losses.svg")
    wandb.log({"quantile_losses": wandb.Image(fig)})

