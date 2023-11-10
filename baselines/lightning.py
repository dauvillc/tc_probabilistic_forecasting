"""
Implements a simple CNN to predict the intensity of a storm one step in advance.
"""
import sys
sys.path.append("./")
import argparse
import torch
import yaml
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from tasks.intensity import intensity_dataset, plot_intensity_bias, plot_intensity_distribution
from data_processing.formats import SuccessiveStepsDataset, datacube_to_tensor
from data_processing.datasets import load_hursat_b1, load_era5_patches
from utils.train_test_split import train_val_test_split
from models.main_structure import StormPredictionModel
from models.cnn3d import CNN3D
from models.variables_projection import VectorProjection3D
from utils.lightning_callbacks import MetricTracker
from utils.utils import hours_to_sincos
from utils.loss_functions import WeightedLoss


def create_model(datacube_size, datacube_channels, num_input_variables, output_length,
                 hidden_channels=8, loss_function=None):
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
    output_length : int
        The number of time steps to predict.
    hidden_channels : int, optional
        The number of channels in the first convolutional layer.
    loss_function : callable, optional
        The loss function to use. If None, the mean squared error is used.
    """
    # Prediction network (3d CNN + Prediction head)
    cnn_model = CNN3D(datacube_size,
                      input_channels=datacube_channels, input_variables=num_input_variables,
                      output_size=future_steps,
                      hidden_channels=args.channels)
    # Projection network (vector projection + 3d CNN)
    projection_model = VectorProjection3D(num_input_variables,
                                          (datacube_channels, ) + datacube_size)
    # Assemble the main structure, built with Lightning
    model = StormPredictionModel(cnn_model, projection_model, loss_function=loss_function)
    return model


if __name__ == "__main__":
    # Some parameters
    input_variables = ['LAT', 'LON', 'HOUR_SIN', 'HOUR_COS']
    output_variables = ['INTENSITY']
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--past_steps", type=int, default=3,
                        help="Number of time steps given as input to the model. Must be >= 3.")
    parser.add_argument("-n", "--prediction_steps", type=int, default=1,
                        help="Number of time steps to predict.")
    parser.add_argument("--channels", type=int, default=8,
                        help="Number of channels in the first convolutional layer.")
    parser.add_argument("--depth" , type=int, default=5,
                        help="Number of convolutional blocks.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs to train the model for.")
    args = parser.parse_args()
    past_steps, future_steps = args.past_steps, args.prediction_steps
    epochs = args.epochs
    if past_steps < 3:
        raise ValueError("The number of past steps must be >= 3.")
    # Load the configuration file
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== DATA LOADING ====== #
    # Load the trajectory forecasting dataset
    all_trajs = intensity_dataset()
    # Add a column with the sin/cos encoding of the hours, which will be used as input
    # to the model
    sincos_hours = hours_to_sincos(all_trajs['ISO_TIME'])
    all_trajs['HOUR_SIN'], all_trajs['HOUR_COS'] = sincos_hours[:, 0], sincos_hours[:, 1]
    # Load the HURSAT-B1 data
    found_storms, hursat_data = load_hursat_b1(all_trajs, use_cache=True, verbose=True)
    # Keep only the storms for which we have HURSAT-B1 data
    all_trajs = all_trajs.merge(found_storms, on=['SID', 'ISO_TIME'])
    # Load the ERA5 patches associated to the dataset
    atmo_patches, surface_patches = load_era5_patches(all_trajs, load_atmo=True)
    # Convert the patches to torch tensors
    era5_patches = datacube_to_tensor(surface_patches)
    hursat_patches = datacube_to_tensor(hursat_data)
    full_patches = torch.cat([era5_patches, hursat_patches], dim=1)


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

        # Create the train and validation datasets
        yield_input_variables = len(input_variables) > 0
        train_dataset = SuccessiveStepsDataset(train_trajs, train_patches, past_steps, future_steps,
                                               input_variables, output_variables,
                                               yield_input_variables=yield_input_variables)
        val_dataset = SuccessiveStepsDataset(val_trajs, val_patches, past_steps, future_steps,
                                             input_variables, output_variables,
                                             yield_input_variables=yield_input_variables)
        # Create the train and validation data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    # Instantiate the train and validation data loaders
    batch_size = 128
    loader_era5, val_loader_era5 = create_dataloaders(era5_patches, batch_size=batch_size)
    loader_hursat, val_loader_hursat = create_dataloaders(hursat_patches, batch_size=batch_size)
    loader_full, val_loader_full = create_dataloaders(full_patches, batch_size=batch_size)
    loaders = {'era5': (loader_era5, val_loader_era5),
               'hursat': (loader_hursat, val_loader_hursat),
               'era5+hursat': (loader_full, val_loader_full)}

    # ====== MODELS CREATION ====== #
    # Create the loss function
    all_intensities = all_trajs['INTENSITY'].values
    loss_function = WeightedLoss(all_intensities, weight_capping_intensity=60)
    # Initialize the models
    patch_size = full_patches.shape[-2:]
    datacube_size = (past_steps,) + patch_size
    # The number of scalar variables the model receives is the number of variables
    # (e.g. 2 for lat/lon) times the number of past steps
    num_input_variables = len(input_variables) * past_steps
    model_era5 = create_model(datacube_size, 4, num_input_variables, future_steps, args.channels, loss_function=loss_function)
    model_hursat = create_model(datacube_size, 1, num_input_variables, future_steps, args.channels, loss_function=loss_function)
    model_full = create_model(datacube_size, 5, num_input_variables, future_steps, args.channels, loss_function=loss_function)
    models = {'era5': model_era5, 'hursat': model_hursat, 'era5+hursat': model_full}

    # ====== MODELS TRAINING ====== #
    # Train the models. Save the train and validation losses
    train_losses, val_losses = {}, {}
    trainers = {}
    for name, model in models.items():
        print(f"Training model {name}...")
        metrics_tracker = MetricTracker()
        trainer = pl.Trainer(accelerator='gpu', precision="bf16-mixed",
                             max_epochs=epochs, callbacks=[metrics_tracker])
        trainer.fit(model, *loaders[name])
        train_losses[name] = metrics_tracker.train_loss
        val_losses[name] = metrics_tracker.val_loss
        # Save the trainer
        trainers[name] = trainer

    # ====== PLOTS ====== #
    # Plot the train and validation losses for each model.
    # The validation losses should be in dashed lines.
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, loss in train_losses.items():
        # Don't plot the first epoch, as it is often an outlier
        ax.plot(loss[1:], label=f"{name} train")
        ax.plot(val_losses[name][1:], '--', label=f"{name} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Train and validation losses for {past_steps} past steps and {future_steps} future steps")
    ax.legend()
    # Save the figure
    plt.tight_layout()
    plt.savefig('figures/baselines/intensity_cnn_losses.png')

    # Retrieve the groundtruth
    any_model_name = list(models.keys())[0]
    y_true = torch.cat([y for _, _, y in loaders[any_model_name][1]], dim=0)

    # Make predictions on the validation set for each model
    val_preds = {}
    for name, model in models.items():
        val_preds[name] = torch.cat(trainers[name].predict(model, loaders[name][1]), dim=0).to(torch.float32)

    # Plot the bias of the predictions for each model
    average_bias = plot_intensity_bias(y_true, val_preds, savepath="figures/baselines/intensity_cnn_bias.png")
    # Plot the distribution of the predictions for each model
    plot_intensity_distribution(y_true, val_preds, savepath="figures/baselines/intensity_cnn_distributions.png")


