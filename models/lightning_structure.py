"""
Implements the training and evaluation tools using the Lightning framework.
"""
import torch
from torch import nn
import pytorch_lightning as pl
from math import prod
from models.encoder import Encoder3d
from models.decoder import Decoder3d
from models.linear import CommonLinearModule, PredictionHead


class StormPredictionModel(pl.LightningModule):
    """
    Implements a LightningModule for the storm prediction task.
    The model takes two inputs (past_variables, past_datacubes):
    - past_variables is a Mapping of str to tensor, with the keys being the names
      of the variables and the values being tensors of shape (N, P * number of variables),
      where P is the number of past steps.
    - past_datacubes is a Mapping of str to tensor, with the keys being the names
        of the datacubes and the values being tensors of shape (N, C, P, H, W).
    The model can have multiple outputs, which can be either vectors or datacubes.
    The outputs are returned as a Mapping of str to tensor, with the keys being
    the task names and the values being prediction tensors.

    Parameters
    ----------
    input_datacube_shape: tuple of int (C, P, H, W)
        The shape of the input datacube.
    num_input_variables: int
        The number of input variables.
    future_steps: int
        The number of future steps to predict.
    tabular_tasks: Mapping of str to Mapping
        The tasks whose targets are vectors, with the keys being the task names and the values
        being the task parameters, including:
        - 'output_variables': list of str
            The names of the output variables for the task.
        - 'output_size': int
            The number of values to predict for the task.
        - 'distrib_obj': Distribution object implementing loss_functio, metrics
            and optionally activation. 
    datacube_tasks: Mapping of str to tuple
        The tasks whose targets are datacubes, with the keys being the task names and the values
        being the shape of the corresponding datacube, as (C, T, H, W).
    cfg: dict
        The configuration dictionary.
    """
    def __init__(self, input_datacube_shape, num_input_variables, tabular_tasks, datacube_tasks, cfg):
        super().__init__()
        self.tabular_tasks = tabular_tasks
        self.datacube_tasks = datacube_tasks
        self.model_cfg = cfg['model_hyperparameters']
        self.training_cfg = cfg['training_settings']
        self.input_datacube_shape = input_datacube_shape
        self.num_input_variables = num_input_variables
        future_steps = cfg['experiment']['future_steps']
        # Create the encoder
        self.encoder = Encoder3d(input_datacube_shape,
                                 self.model_cfg['base_block'],
                                 conv_blocks=self.model_cfg['encoder_depth'],
                                 hidden_channels=self.model_cfg['encoder_channels'])
        # Create the common linear module
        # The output size of the CLM is (T * h * w * c) where T is the number of future steps,
        # h and w are the width of the encoded latent space and c is an arbitrary number of channels.
        # This is so that the vector can be reshaped into a tensor of shape (T, h, w, c) and fed
        # into the decoder.
        encoder_output_shape = self.encoder.output_shape
        self.clm_output_channels = self.model_cfg['clm_output_channels']
        clm_output_size = future_steps * prod(encoder_output_shape[2:]) * self.clm_output_channels
        self.common_linear_model = CommonLinearModule(self.encoder.output_shape,
                                                      num_input_variables * future_steps,
                                                      clm_output_size)
        # Create the prediction heads
        self.prediction_heads = nn.ModuleDict({})
        self.loss_functions = {}
        for task, task_params in tabular_tasks.items():
            # Create the prediction head
            self.prediction_heads[task] = PredictionHead(clm_output_size, task_params['output_size'], future_steps)

        # Create the decoder if needed
        if len(datacube_tasks) > 0:
            self.decoder_input_shape = (self.clm_output_channels, future_steps, *encoder_output_shape[2:])
            # The output of the decoder is all target datacubes concatenated along the channel dimension
            decoder_output_channels = sum([datacube_shape[0] for datacube_shape in datacube_tasks.values()])
            self.decoder = Decoder3d(self.clm_output_channels, decoder_output_channels,
                                     self.model_cfg['base_block'],
                                     self.model_cfg['encoder_depth'])

    def training_step(self, batch, batch_idx):
        """
        Implements a training step.
        """
        past_variables, past_datacubes, future_variables, future_datacubes = batch
        predictions = self.forward(past_variables, past_datacubes)
        # Compute the indivual losses for each task
        losses = {}
        for task, task_params in self.tabular_tasks.items():
            losses[task] = task_params['distrib_obj'].loss_function(predictions[task], future_variables[task])
            self.log(f"train_loss_{task}", losses[task], on_step=False, on_epoch=True)
        # Compute the losses for the datacube tasks
        for task, task_params in self.datacube_tasks.items():
            losses[task] = nn.MSELoss()(predictions[task], future_datacubes[task])
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
        predictions = self.forward(past_variables, past_datacubes)
        # Compute the indivual losses for each tabular task
        losses = {}
        for task, task_params in self.tabular_tasks.items():
            # Denormalize the predictions using the task-specific denormalization function
            predictions[task] = task_params['distrib_obj'].denormalize(predictions[task], task)
            # Compute the loss in the original scale
            losses[task] = task_params['distrib_obj'].loss_function(predictions[task], future_variables[task])
            self.log(f"val_loss_{task}", losses[task], on_step=False, on_epoch=True)
        # Compute the losses for the datacube tasks
        for task, task_params in self.datacube_tasks.items():
            #TODO denormalize the predictions
            losses[task] = nn.MSELoss()(predictions[task], future_datacubes[task])
            self.log(f"val_loss_{task}", losses[task], on_step=False, on_epoch=True)
        # Compute the total loss
        total_loss = sum(losses.values())
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Compute the metrics for each task
        for task, task_params in self.tabular_tasks.items():
            for metric_name, metric in task_params['distrib_obj'].metrics.items():
                # Compute the metric in the original scale
                metric_value = metric(predictions[task], future_variables[task])
                self.log(f"val_{metric_name}_{task}", metric_value, on_step=False, on_epoch=True)

        return total_loss

    def predict_step(self, batch, batch_idx=0, dataloader_idx=None):
        """
        Implements a prediction step.
        """
        past_variables, past_datacubes, future_variables, future_datacubes = batch
        predictions = self.forward(past_variables, past_datacubes)
        return predictions
 
    def forward(self, past_variables, past_datacubes):
        """
        Parameters
        ----------
        past_variables: Mapping of str to tensor
            The past variables, with the keys being the names of the variables and the values
            being batches of shape (N, P * number of variables).
        past_datacubes: Mapping of str to tensor
            The past datacubes, with the keys being the names of the datacubes and the values
            being batches of shape (N, C, P, H, W).

        Returns
        -------
        Mapping of str to tensor
            The predictions, with the keys being the task names and the values being batches
            of predicted tensors.
        """
        # Concatenate the past datacubes into a single tensor along the channel dimension
        past_datacubes = torch.cat(list(past_datacubes.values()), dim=1)
        # Encode the past datacubes into a latent space
        latent_space = self.encoder(past_datacubes)
        # Apply the common linear model to the latent space and the past variables
        latent_space = self.common_linear_model(latent_space, past_variables)
        # Apply the prediction heads
        predictions = {}
        for task, task_params in self.tabular_tasks.items():
            predictions[task] = self.prediction_heads[task](latent_space)
            # Check if there is an activation function specific to the distribution
            if hasattr(task_params['distrib_obj'], 'activation'):
                predictions[task] = task_params['distrib_obj'].activation(predictions[task])
        # Apply the decoder if needed
        if len(self.datacube_tasks) > 0:
            # Reshape the latent space from a vector to a tensor of shape (T, h, w, c)
            latent_space = latent_space.view(latent_space.shape[0], *self.decoder_input_shape)
            datacube_preds = self.decoder(latent_space)
            # The target datacubes are concatenated along the channel dimension, retrieve them
            # individually
            start_channel = 0 
            for task, datacube_shape in self.datacube_tasks.items():
                end_channel = start_channel + datacube_shape[0]
                predictions[task] = datacube_preds[:, start_channel:end_channel]
                start_channel = end_channel
        return predictions

    def configure_optimizers(self):
        """
        Configures the optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.training_cfg['initial_lr'],
                                     weight_decay=self.training_cfg['weight_decay'])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=self.training_cfg['epochs'])
        return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                }
