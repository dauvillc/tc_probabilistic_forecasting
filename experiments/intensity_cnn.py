"""
Implements a simple CNN to predict the intensity of a storm one step in advance.
"""
import sys
sys.path.append("./")
import argparse
import torch
import yaml
import matplotlib.pyplot as plt
from tasks.intensity import intensity_dataset, plot_intensity_bias, plot_intensity_distribution
from data_processing.formats import SuccessiveStepsDataset, datacube_to_tensor
from data_processing.datasets import load_hursat_b1, load_era5_patches
from utils.train_test_split import train_val_test_split
from baselines.train import train
from models.cnn3d import CNN3D


if __name__ == "__main__":
    # Some parameters
    input_variables = ['LAT', 'LON']
    output_variables = ['INTENSITY']
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--past_steps", type=int, default=3,
                        help="Number of time steps given as input to the model. Must be >= 3.")
    parser.add_argument("-n", "--prediction_steps", type=int, default=1,
                        help="Number of time steps to predict.")
    parser.add_argument("--channels", type=int, default=8,
                        help="Number of channels in the first convolutional layer.")
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
    # Load the HURSAT-B1 data
    found_storms, hursat_data = load_hursat_b1(all_trajs, use_cache=True, verbose=True)
    # Keep only the storms for which we have HURSAT-B1 data
    all_trajs = all_trajs.merge(found_storms, on=['SID', 'ISO_TIME'])
    # Load the ERA5 patches associated to the dataset
    atmo_patches, surface_patches = load_era5_patches(all_trajs, load_atmo=False)
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

    # ====== MODELS TRAINING ====== #
    # Initialize the models
    patch_size = era5_patches.shape[-2:]
    input_size = (past_steps,) + patch_size
    # The number of scalar variables the model receives is the number of variables
    # (e.g. 2 for lat/lon) times the number of past steps
    num_input_variables = len(input_variables) * past_steps
    model_era5 = CNN3D(input_size,
                       input_channels=3, input_variables=num_input_variables, output_size=future_steps,
                       hidden_channels=args.channels).to(device)
    model_hursat = CNN3D(input_size,
                         input_channels=1, input_variables=num_input_variables, output_size=future_steps,
                         hidden_channels=args.channels).to(device)
    model_full = CNN3D(input_size, input_channels=4, input_variables=num_input_variables, output_size=future_steps,
                       hidden_channels=args.channels).to(device)
    models = {'era5': model_era5, 'hursat': model_hursat, 'era5+hursat': model_full}

    # Instantiate the train and validation data loaders
    loader_era5, val_loader_era5 = create_dataloaders(era5_patches)
    loader_hursat, val_loader_hursat = create_dataloaders(hursat_patches)
    loader_full, val_loader_full = create_dataloaders(full_patches)
    loaders = {'era5': (loader_era5, val_loader_era5),
               'hursat': (loader_hursat, val_loader_hursat),
               'era5+hursat': (loader_full, val_loader_full)}

    # Train and evaluate the models, and save their train and validation losses
    # in a dictionary
    train_losses, val_losses = {}, {}
    for model in models.keys():
        print(f"Training model {model}...")
        train_losses[model], val_losses[model] = train(models[model], *loaders[model],
                                                       epochs=epochs, device=device)

    # Plot the train and validation losses in a single figure.
    # Plot the validation losses as a dashed line.
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in train_losses.keys():
        ax.plot(train_losses[key], label=f"{key} train")
        ax.plot(val_losses[key], label=f"{key} val", linestyle='--')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train and validation losses")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/baselines/intensity_cnn_losses.png")

    # Make predictions on the validation set for each model in models
    with torch.no_grad():
        y_true, y_pred = [], {}
        # We'll only fill the groundtruth once, since it's the same for all models
        y_true_filled = False
        for name, model in models.items():
            y_pred[name] = []
            for past_vars, past_images, future_images in loaders[name][1]:
                x1, x2, y = past_images.to(device), past_vars.to(device), future_images.to(device)
                y_pred[name].append(model(x1, x2).cpu())
                if not y_true_filled:
                    y_true.append(y.cpu())
            if not y_true_filled:
                y_true = torch.cat(y_true, dim=0)
                y_true_filled = True
            y_pred[name] = torch.cat(y_pred[name], dim=0)

        # plot the distribution of the bias of the intensity prediction for each
        # predicted time step.
        bias = plot_intensity_bias(y_true, y_pred, savepath="figures/baselines/intensity_cnn_bias.png")
        # Plot the distributions themselves
        plot_intensity_distribution(y_true, y_pred, savepath="figures/baselines/intensity_cnn_distributions.png")
        print("Average bias:")
        print(bias)

