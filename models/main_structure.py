"""
Implements the training and evaluation tools using the Lightning framework.
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class StormPredictionModel(pl.LightningModule):
    """
    Implements a LightningModule for the storm prediction task.
    The model takes two types of input:
    - A vector of shape (N, D * n_input_vars), containing the
      past values of the scalar variables (e.g. lat, lon, intensity, etc.)
    - One or multiple datacubes of shape (N, C, D, H, W), containing 3D (2D + time)
      data (e.g. reanalysis, satellite, etc.)

    Parameters
    ----------
    prediction_model : torch Module
        The model to use for the prediction task. Should take as input a datacube
        of shape (N, C, D, H, W) and return a tensor of shape (N, n_predicted_steps).
    projection_model: torch.Module
        The model to use to project the scalar variables into a datacube.
        Should take as input a tensor of shape (N, D * n_input_vars)
        and return a tensor of shape (N, C', D, H, W).
    """
    def __init__(self, prediction_model, projection_model):
        super().__init__()
        self.prediction_model = prediction_model
        self.projection_model = projection_model

    def training_step(self, batch, batch_idx):
        """
        Implements a training step.
        """
        past_variables, past_data, target_variables = batch
        prediction = self.forward(past_variables, past_data)
        # Compute the loss
        loss = F.mse_loss(prediction, target_variables)
        # Log the loss
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Implements a validation step.
        """
        past_variables, past_data, target_variables = batch
        prediction = self.forward(past_variables, past_data)
        # Compute the loss
        loss = F.mse_loss(prediction, target_variables)
        # Log the loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        """
        Implements a prediction step.
        """
        past_variables, past_data, target_variables = batch
        prediction = self.forward(past_variables, past_data)
        return prediction

    def forward(self, past_variables, past_data):
        """
        Makes a prediction for the given input.
        """
        # Project the variables into a datacube
        projected_vars = self.projection_model(past_variables)
        # Make the prediction on the sum of the past data and the projected variables
        prediction = self.prediction_model(past_data + projected_vars)
        return prediction

    def configure_optimizers(self):
        """
        Configures the optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                }
