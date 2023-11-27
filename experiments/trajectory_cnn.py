"""
Clément Dauvilliers - 2023 10 17
Implements a simple CNN to predict the trajectory of a storm one step in advance.
"""
import sys
sys.path.append("./")
import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt
import datetime as dt
from tqdm import tqdm
from tasks.trajectory import trajectory_dataset
from data_processing import load_era5_patches
from data_processing.format_conversion import era5_patches_to_tensors
from utils.train_test_split import train_val_test_split
from utils.datacube import select_sid_time


class TrajectoryCNN(nn.Module):
    """
    A simple CNN to predict the trajectory of a storm one step in advance.
    """
    def __init__(self, input_channels=69):
        """
        Parameters
        ----------
        input_channels : int, optional
            Number of input channels. The default is 69 (5 atmospheric variables
            at 13 levels + 4 surface variables).
        """
        super().__init__()
        # Add a batch normalization layer at the beginning
        self.batchnorm = nn.BatchNorm2d(input_channels)
        # Convolutional blocks:
        # Each block is composed of 2 convolutional layers with a kernel size
        # of 3 and a padding of 1, followed by a batch normalization layer.
        self.conv1_1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1) # 11x11 -> 11x11
        # The second layer of the block has a stride of 2, instead of using
        # max pooling
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2) # 11x11 -> 5x5
        self.conv1_batchnorm = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # 5x5 -> 5x5
        self.conv2_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2) # 5x5 -> 3x3
        self.conv2_batchnorm = nn.BatchNorm2d(128)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch tensor of dimensions (N, C, H, W)
            Input batch of N patches of dimensions (C, H, W).

        Returns
        -------
        torch tensor of dimensions (N, 2)
            Output batch of N predicted displacements.
        """
        # Input normalization
        x = self.batchnorm(x)
        # Convolutional blocks
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_batchnorm(x)
        x = nn.functional.selu(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_batchnorm(x)
        x = nn.functional.selu(x)
        # Fully connected layers
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = nn.functional.selu(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # Load the configuration file
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== DATA LOADING ====== #
    # Load the trajectory forecasting dataset
    all_trajs = trajectory_dataset()
    # Specify the start and end dates: only keep the trajectories that
    # start between these dates
    start_date = dt.datetime(2000, 1, 1)
    end_date = dt.datetime(2021, 12, 31)

    # Load the ERA5 patches (atmospheric and surface variables)
    atmo_patches, surface_patches = load_era5_patches(start_date, end_date)
    # Ensure that the patches correspond to the trajectories
    atmo_patches = select_sid_time(atmo_patches, all_trajs['SID'].values, all_trajs['ISO_TIME'].values)
    surface_patches = select_sid_time(surface_patches, all_trajs['SID'].values, all_trajs['ISO_TIME'].values)

    # Convert the patches to torch tensors
    atmo_patches = era5_patches_to_tensors(atmo_patches)
    surface_patches = era5_patches_to_tensors(surface_patches)
    # Stack the atmospheric and surface patches along the channel dimension
    patches = torch.cat((atmo_patches, surface_patches), dim=1)

    # Split the dataset into train, validation and test sets
    train_index, val_index, test_index = train_val_test_split(all_trajs,
                                                              train_size=0.6,
                                                              val_size=0.2,
                                                              test_size=0.2)
    train_trajs = all_trajs.iloc[train_index]
    val_trajs = all_trajs.iloc[val_index]
    test_trajs = all_trajs.iloc[test_index]

    # Split the patches into train, validation and test sets
    train_patches = patches[train_index]
    val_patches = patches[val_index]
    test_patches = patches[test_index]

    print(f"Number of trajectories in the training set: {len(train_trajs)}")
    print(f"Number of trajectories in the validation set: {len(val_trajs)}")
    print(f"Number of trajectories in the test set: {len(test_trajs)}")

    # Create the train and validation datasets and DataLoaders
    train_target = torch.tensor(train_trajs[['DELTA_LAT', 'DELTA_LON']].to_numpy(),
                                dtype=torch.float32)
    val_target = torch.tensor(val_trajs[['DELTA_LAT', 'DELTA_LON']].to_numpy(),
                              dtype=torch.float32)
    batch_size = 128
    # Apply a normalizing transformation to the patches
    train_dataset = torch.utils.data.TensorDataset(train_patches, train_target)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(val_patches, val_target)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ====== MODEL TRAINING ====== #
    # Initialize the model
    model = TrajectoryCNN().to(device)
    # Initialize the optimizer and the step LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # Initialize the loss function
    loss_fn = nn.MSELoss()
    n_epochs = 50

    # Train the model and evaluate it on the validation set at the end of each
    # epoch. Save the train and validation losses for each epoch.
    train_losses, val_losses = [], []
    for epoch in range(n_epochs):
        epoch_train_loss, epoch_val_loss = 0, 0
        # Train over the epoch
        model.train()
        for (x, y) in tqdm(train_loader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            # Save the train loss for the epoch
            epoch_train_loss += loss.item()
        # Compute the average train loss for the epoch
        train_losses.append(epoch_train_loss / len(train_loader))

        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            for (x, y) in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                # Save the validation loss for the epoch
                epoch_val_loss += loss.item()
            # Compute the average validation loss for the epoch
            val_losses.append(epoch_val_loss / len(val_loader))

        # Print the train and validation losses for the epoch
        # in a compact format
        print(f"Epoch {epoch + 1}/{n_epochs} | "
              f"Train loss: {train_losses[-1]:.4f} | "
              f"Validation loss: {val_losses[-1]:.4f}")

        # Update the learning rate
        scheduler.step()

    # Plot the train and validation losses in 
    plt.figure()
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Save the figure under figures/baselines/
    plt.savefig('figures/baselines/trajectory_cnn_losses.png')
