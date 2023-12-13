"""
Implements the training and evaluation tools using the Lightning framework.
"""
import torch
from torch import nn
import pytorch_lightning as pl
from models.encoder import Encoder3d
from models.linear import CommonLinearModule, PredictionHead


class StormPredictionModel(pl.LightningModule):
    """
    Implements a LightningModule for the storm prediction task.
    The model takes two inputs (past_variables, past_datacubes):
    - past_variables is a Mapping of str to tensor, with the keys being the names
      of the variables and the values being tensors of shape (N, P).
    - past_datacubes is a Mapping of str to tensor, with the keys being the names
        of the datacubes and the values being tensors of shape (N, C, P, H, W).
    The model can have multiple outputs, which can be either vectors or datacubes.
    The outputs are returned as a Mapping of str to tensor, with the keys being
    the task names and the values being prediction tensors.

    Parameters
    ----------
    datacube_shape: tuple of int (C, D, H, W)
        The shape of the input datacube.
    num_input_variables: int
        The number of input variables.
    future_steps: int
        The number of future steps to predict.
    tasks: Mapping of str to Mapping
        The tasks to perform, with the keys being the task names and the values
        being the task parameters, including:
        - 'output_variables': list of str
            The names of the output variables for the task.
        - 'output_size': int
            The number of values to predict for the task.
        - 'loss_function': callable
            The loss function for the task.
    """
    def __init__(self, datacube_shape, num_input_variables, future_steps, tasks):
        super().__init__()
        self.tabular_tasks = tasks
        self.datacube_shape = datacube_shape
        self.num_input_variables = num_input_variables
        # Create the encoder
        self.encoder = Encoder3d(datacube_shape) 
        # Create the common linear module
        self.common_linear_model = CommonLinearModule(self.encoder.output_shape,
                                                      num_input_variables * future_steps,
                                                      128)
        # Create the prediction heads and loss functions
        self.prediction_heads = nn.ModuleDict({})
        self.loss_functions = {}
        for task, task_params in tasks.items():
            # Create the prediction head
            self.prediction_heads[task] = PredictionHead(128, task_params['output_size'], future_steps)
            # Create the loss function
            self.loss_functions[task] = task_params['loss_function']

    def training_step(self, batch, batch_idx):
        """
        Implements a training step.
        """
        past_variables, past_datacubes, future_variables, future_datacubes = batch
        predictions = self.forward(batch, batch_idx)
        # Compute the indivual losses for each task
        losses = {}
        for task in self.tabular_tasks:
            losses[task] = self.loss_functions[task](predictions[task], future_variables[task])
            self.log(f"train_loss_{task}", losses[task], on_step=False, on_epoch=True)
        # Compute the total loss
        total_loss = sum(losses.values())
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Implements a validation step.
        """
        past_variables, past_datacubes, future_variables, future_datacubes = batch
        predictions = self.forward(batch, batch_idx)
        # Compute the indivual losses for each task
        losses = {}
        for task in self.tabular_tasks:
            # Denormalize the predictions using the task-specific denormalization function
            predictions[task] = self.tabular_tasks[task]['denormalize'](predictions[task], task)
            # Compute the loss in the original scale
            losses[task] = self.loss_functions[task](predictions[task], future_variables[task])
            self.log(f"val_loss_{task}", losses[task], on_step=False, on_epoch=True)
        # Compute the total loss
        total_loss = sum(losses.values())
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss
 
    def forward(self, batch, batch_idx):
        """
        Parameters
        ----------
        batch: tuple (past_vars, past_datacubes, future_vars, future_datacubes)
            Where each element is a Mapping of str to tensor.
        batch_idx: int
            The index of the batch.
        """
        past_variables, past_datacubes, future_variables, future_datacubes = batch
        # Concatenate the past datacubes into a single tensor along the channel dimension
        past_datacubes = torch.cat(list(past_datacubes.values()), dim=1)
        # Encode the past datacubes into a latent space
        latent_space = self.encoder(past_datacubes)
        # Apply the common linear model to the latent space and the past variables
        latent_space = self.common_linear_model(latent_space, past_variables)
        # Apply the prediction heads
        predictions = {}
        for task in self.tabular_tasks:
            predictions[task] = self.prediction_heads[task](latent_space)
        return predictions

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
