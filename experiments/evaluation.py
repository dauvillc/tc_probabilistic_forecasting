"""
Evaluates a model taken from a W&B run. All plots are uploaded to W&B and
saved locally.
"""
import sys
sys.path.append("./")
import argparse
import numpy as np
import wandb
import torch
import pytorch_lightning as pl
from pathlib import Path
from models.main_structure import StormPredictionModel
from data_processing.assemble_experiment_dataset import load_dataset
from metrics.probabilistic import metric_per_threshold
from plotting.quantiles import plot_quantile_losses
from plotting.distributions import plot_data_distribution
from plotting.probabilistic import plot_metric_per_threshold
from experiments.training import create_model, create_output_distrib


if __name__ == "__main__":
    pl.seed_everything(42)
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
    runs = api.runs("arches/tc_prediction", filters={"config.experiment.name": args.name})
    if len(runs) == 0:
        raise ValueError(f"No runs with name {args.name} were found.")
    # If several runs have the same name, we'll use any but print a warning
    if len(runs) > 1:
        print(f"WARNING: Several runs with name {args.name} were found. Using the first one.")
    # Get the run id
    evaluated_run = runs[0]
    run_id = evaluated_run.id
    # We can now retrieve the config of the run
    cfg = evaluated_run.config
    experiment_cfg = cfg["experiment"]
    model_cfg = cfg["model_hyperparameters"]
    past_steps, future_steps = experiment_cfg["past_steps"], experiment_cfg["future_steps"]

    # ====== DATA LOADING ====== #
    # Retrieve the input type from the run config
    input_data = experiment_cfg["input_data"]
    train_dataset, val_dataset, train_loader, val_loader = load_dataset(cfg, input_variables, output_variables)

    # ===== OUTPUT DISTRIBUTION RECONSTRUCTION ====== #
    # Create the output distribution
    distrib_name = experiment_cfg["distribution"]
    distrib = create_output_distrib(distrib_name, train_dataset)
    
    # ====== MODEL RECONSTRUCTION ====== #
    # Create the model architecture
    # We need to re-create the architecture, as it is not created within the LightningModule.
    # Once we have recreated it, we can load the weights from the checkpoint.
    patch_size = train_dataset.patch_size()
    datacube_size = (past_steps,) + patch_size
    datacube_channels = train_dataset.datacube_channels()
    num_input_variables = len(input_variables) * past_steps
    model = create_model(datacube_size, datacube_channels, num_input_variables,
                         future_steps, distrib.n_parameters, hidden_channels=model_cfg['hidden_channels'])
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
    wandb.log(cfg)
    wandb.log(distrib.hyperparameters())

    # ====== EVALUATION ====== #
    # Make predictions on the validation set
    y_true = val_dataset.future_target
    trainer = pl.Trainer(accelerator="gpu", max_epochs=1)
    pred = torch.cat([p.cpu().detach() for p in trainer.predict(model, val_loader)], dim=0)

    # Create a directory in the figures folder specific to this experiment
    figpath = Path(f"figures/{experiment_cfg['name']}")
    figpath.mkdir(exist_ok=True)

    # Plot the distribution of the true values
    fig = plot_data_distribution(y_true, quantiles=[0.1, 0.5, 0.75, 0.9, 0.95],
                                 savepath=f"{figpath}/true_distribution.svg")
    wandb.log({"true_distribution": wandb.Image(fig)})

    # Compute the CRPS
    # As the max in the validation dataset might be higher, we'll apply some margin
    crps = distrib.metrics["CRPS"](pred, y_true)
    print("Average CRPS:", crps)
    wandb.log({"avg_crps": crps})
    
    # We'll now compute several metrics per threshold:
    # Given a true value y and its associated predicted distribution F,
    # we compute the metric L(y, F^{-1}(u)) for each threshold u.
    thresholds = np.linspace(0, 1, 22)[1:-1]
    # These metrics will be computed over the whole validation set, but also
    # for subsets consisting of increasingly extreme values
    y_true_quantiles = [0.5, 0.75, 0.9, 0.95]
    # Get the inverse of the CDF, specific to the distribution
    inverse_CDF = distrib.inverse_cdf
    # Compute the metrics
    metrics = ["bias", "mae", "rmse"]
    for metric in metrics:
        metric_df = metric_per_threshold(metric, y_true, pred, inverse_CDF, thresholds,
                                         y_true_quantiles)
        # Plot the metric per threshold
        fig = plot_metric_per_threshold(metric, metric_df, y_true_quantiles,
                                        save_path=f"{figpath}/{metric}_per_threshold.svg")
        wandb.log({f"{metric}_per_threshold": wandb.Image(fig)})
        # Log the metric per threshold to wandb
        wandb.log({f"{metric}_per_threshold_table": wandb.Table(dataframe=metric_df)})

    # DISTRIBUTION-SPECIFIC PLOTS
    if distrib_name in ["quantile_composite", "qc"]:
        # Plot the quantile losses and log them to wandb
        fig = plot_quantile_losses({"model": pred}, y_true, distrib.quantiles,
                                    savepath=f"{figpath}/quantile_losses.svg")
        wandb.log({"quantile_losses": wandb.Image(fig)})

