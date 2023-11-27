"""
Implements the basic training loop for a CNN model.
"""
import torch
import torch.nn as nn
from tqdm import tqdm


def train(model, train_loader, val_loader, device=None, epochs=50,
          optimizer=None, lr_scheduler=None):
    """
    Trains the model for the specified number of epochs, using the specified
    optimizer and learning rate scheduler.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    train_loader : torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    device : torch.device
        If None, defaults to torch.device("cuda") if available, else
        torch.device("cpu").
    epochs: int
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler

    Returns
    -------
    train_losses : list of float
        The average training loss for each epoch.
    val_losses : list of float
        The average validation loss for each epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the loss function
    loss_fn = nn.MSELoss()
    # Initialize the optimizer and the step LR scheduler
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if lr_scheduler is None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # Train the model and evaluate it on the validation set at the end of each 
    # epoch. Save the train and validation losses.
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        tr_loss, val_loss = 0, 0
        # Train over the epoch
        model.train()
        for past_vars, past_images, future_vars in tqdm(train_loader):
            x1, x2, y = past_images.to(device), past_vars.to(device), future_vars.to(device)
            optimizer.zero_grad()
            y_pred = model(x1, x2)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            # Save the train loss for the epoch
            tr_loss += loss.item() / x1.shape[0]
        # Compute the average train loss for the epoch
        train_losses.append(tr_loss / len(train_loader))
        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            for past_vars, past_images, future_vars in val_loader:
                x1, x2, y = past_images.to(device), past_vars.to(device), future_vars.to(device)
                y_pred = model(x1, x2)
                loss = loss_fn(y_pred, y)
                # Save the validation loss for the epoch
                val_loss += loss.item() / x1.shape[0]
            # Compute the average validation loss for the epoch
            val_losses.append(val_loss / len(val_loader))
        # Print the train and validation losses for the epoch
        # in a compact format
        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train loss: {train_losses[-1]:.4f} | "
              f"Validation loss: {val_losses[-1]:.4f}")
        # Update the learning rate
        lr_scheduler.step()
    return train_losses, val_losses
