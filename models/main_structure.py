"""
Implements the training and evaluation tools using the Lightning framework.
"""
import torch
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
    loss_function : callable
        The loss function to use.
    metrics: Mapping of str to callable, optional
        The metrics to track. The keys are the names of the metrics, and the values
        are functions that take as input the output of the model and the target,
        and return a scalar.
    """
    def __init__(self, prediction_model, projection_model, loss_function, metrics=None):
        super().__init__()
        self.prediction_model = prediction_model
        self.projection_model = projection_model
        self.loss_function = loss_function
        self.metrics = metrics if metrics is not None else {}

    def training_step(self, batch, batch_idx):
        """
        Implements a training step.
        """
        past_variables, past_datacube, future_variables, future_datacube = batch
        future_variables = future_variables['INTENSITY']
        prediction = self.forward(past_variables, past_datacube)
        # Compute the loss
        loss = self.loss_function(prediction, future_variables)
        # Log the loss
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # Compute the metrics
        for name, metric in self.metrics.items():
            self.log("train_" + name, metric(prediction, future_variables),
                     on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Implements a validation step.
        """
        past_variables, past_datacube, future_variables, future_datacube = batch
        future_variables = future_variables['INTENSITY']
        prediction = self.forward(past_variables, past_datacube)
        # Compute the loss
        loss = self.loss_function(prediction, future_variables)
        # Log the loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # Compute the metrics
        for name, metric in self.metrics.items():
            self.log("val_" + name, metric(prediction, future_variables),
                     on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        """
        Implements a prediction step.
        """
        past_variables, past_datacube, future_variables, future_datacube = batch
        prediction = self.forward(past_variables, past_datacube)
        return prediction

    def forward(self, past_variables, past_data):
        """
        Makes a prediction for the given input.
        """
        # Concatenate the past variables into a single tensor
        past_variables = torch.cat(list(past_variables.values()), dim=1)
        # Project the variables into a datacube
        projected_vars = self.projection_model(past_variables)
        # Concatenate all the datacubes into a single tensor
        past_data = torch.cat(list(past_data.values()), dim=1)
        # Make the prediction on the sum of the past data and the projected variables
        prediction = self.prediction_model(past_data + projected_vars)
        return prediction

    def configure_optimizers(self):
        """
        Configures the optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                }
