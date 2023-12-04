"""
Evaluates a model taken from a W&B run. All plots are uploaded to W&B and
saved locally.
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
from metrics.quantiles import QuantilesCRPS, Quantiles_inverse_eCDF
from metrics.probabilistic import mae_per_threshold
from plotting.quantiles import plot_quantile_losses
from plotting.distributions import plot_data_distribution


if __name__ == "__main__":
    # Some parameters
    input_variables = ['LAT', 'LON', 'HOUR_SIN', 'HOUR_COS']
    output_variables = ['INTENSITY']
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True,
                        help="The name of the experiment to evaluate.")
    args = parser.parse_args()

    # ====== WANDB INITIALIZATION ====== #
    # Initialize the W+B logger and retrieve the run that led to the model being
    # evaluated
    run = wandb.init(project="tc_prediction", name="eval_" + args.name, job_type="eval")
    api = wandb.Api()
    # Search for all runs with the same name
    runs = api.runs("arches/tc_prediction", filters={"config.name": args.name})
    if len(runs) == 0:
        raise ValueError(f"No runs with name {args.name} were found.")
    # If several runs have the same name, we'll use any but print a warning
    if len(runs) > 1:
        print(f"WARNING: Several runs with name {args.name} were found. Using the first one.")
    # Get the run id
    evaluated_run = runs[0]
    run_id = evaluated_run.id
    # We can now retrieve the config of the run
    for key, value in evaluated_run.config.items():
        if key not in args:
            vars(args)[key] = value
    quantiles = args.quantiles
    past_steps, future_steps = args.past_steps, args.future_steps

    # ====== DATA LOADING ====== #
    # Retrieve the input type from the run config
    input_data = evaluated_run.config["input_data"]
    args.input_data = input_data
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
    
    # Load the weights from the best checkpoint
    artifact = run.use_artifact("model-" + run_id + ":best", type="model")
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

    # Create a directory in the figures folder specific to this experiment
    figpath = Path(f"figures/{args.name}")
    figpath.mkdir(exist_ok=True)

    # Plot the distribution of the true values
    fig = plot_data_distribution(y_true, quantiles=quantiles, savepath=f"{figpath}/true_distribution.svg")
    wandb.log({"true_distribution": wandb.Image(fig)})

    # Compute the CRPS
    crps_computer = QuantilesCRPS(quantiles, 0, 100)
    crps = crps_computer(pred, y_true)
    print("Average CRPS:", crps)
    wandb.log({"avg_crps": crps})

    # Plot the quantile losses and log them to wandb
    fig = plot_quantile_losses({"model": pred}, y_true, quantiles,
                                savepath=f"{figpath}/quantile_losses.svg")
    wandb.log({"quantile_losses": wandb.Image(fig)})

    # Compute the MAE between the true value and the prediction of the model at a given
    # point of the CDF
    thresholds = np.linspace(0, 1, 11)
    inverse_CDF = Quantiles_inverse_eCDF(quantiles, min_val=0, max_val=100)
    maes = mae_per_threshold(y_true, pred, inverse_CDF, thresholds)
    print("MAE per threshold:")
    print(maes)
    # Log the thresholds and the MAEs to a W&B table
    table = wandb.Table(columns=["threshold", "MAE"], data=[[t, m] for t, m in zip(thresholds, maes)])
    wandb.log({"mae_per_threshold": table})

