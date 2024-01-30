"""
Implements the training and evaluation tools using the Lightning framework.
"""
import torch
from torch import nn
import pytorch_lightning as pl
from models.encoder import Encoder3d
from models.decoder import Decoder3d
from models.linear import CommonLinearModule, PredictionHead, MultivariatePredictionHead


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
        - 'distrib_obj': Distribution object implementing loss_function, metrics
            and optionally activation. 
    datacube_tasks: Mapping of str to tuple
        The tasks whose targets are datacubes, with the keys being the task names and the values
        being:
        - 'output_channels: int
            The number of channels of the output datacube.
        - 'distrib_obj': Distribution object implementing loss_function and metrics.
    train_dataset: SuccessiveStormsDataset
        Pointer to the training dataset.
    val_dataset: SuccessiveStormsDataset
        Pointer to the validation dataset. Used to denormalize the predictions.
    cfg: dict
        The configuration dictionary.
    """
    def __init__(self, input_datacube_shape, num_input_variables, tabular_tasks, datacube_tasks,
                 train_dataset, val_dataset, cfg):
        super().__init__()
        self.tabular_tasks = tabular_tasks
        self.datacube_tasks = datacube_tasks
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
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
        self.common_linear_model = CommonLinearModule(self.encoder.output_shape,
                                                      future_steps,
                                                      num_input_variables * future_steps,
                                                      cfg['model_hyperparameters']['clm_reduction_factor'])
        # Create the prediction heads
        self.prediction_heads = nn.ModuleDict({})
        self.loss_functions = {}
        for task, task_params in tabular_tasks.items():
            # Create the prediction head. If the task is to predict a distribution at each
            # time step, use PredictionHead. Otherwise, use MultivariatePredictionHead.
            if task_params['distrib_obj'].is_multivariate:
                self.prediction_heads[task] = MultivariatePredictionHead(self.common_linear_model.output_size,
                                                                         task_params['output_size'])
            else:   
                self.prediction_heads[task] = PredictionHead(self.common_linear_model.output_size,
                                                             task_params['output_size'], future_steps)

        # Create the decoder if needed
        if len(datacube_tasks) > 0:
            # The output of the decoder is all target datacubes concatenated along the channel dimension
            decoder_output_channels = sum([task['output_channels'] for task in datacube_tasks.values()])
            self.decoder = Decoder3d(self.encoder, decoder_output_channels,
                                     self.model_cfg['base_block'],
                                     self.model_cfg['encoder_depth'])

    
    def compute_losses(self, batch, train_or_val='train'):
        """
        Computes the losses for a single batch (subfunction of training_step
        and validation_step).
        """
        past_variables, past_datacubes, future_variables, future_datacubes = batch
        predictions = self.forward(past_variables, past_datacubes)
        # Compute the indivual losses for each task
        losses = {}
        for task, task_params in self.tabular_tasks.items():
            losses[task] = task_params['distrib_obj'].loss_function(predictions[task], future_variables[task])
            self.log(f"{train_or_val}_loss_{task}", losses[task], on_step=False, on_epoch=True)
        # Compute the losses for the datacube tasks
        for task, task_params in self.datacube_tasks.items():
            losses[task] = nn.MSELoss()(predictions[task], future_datacubes[task])
            self.log(f"{train_or_val}_loss_{task}", losses[task], on_step=False, on_epoch=True)
        # Compute the total loss
        # In the case of a single task, the total loss is the loss of the trained task
        if self.training_cfg['trained_task'] is not None:
            total_loss = losses[self.training_cfg['trained_task']]
        else:
            total_loss = sum(losses.values())
        self.log(f"{train_or_val}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss


    def training_step(self, batch, batch_idx):
        """
        Implements a training step.
        """
        return self.compute_losses(batch, train_or_val='train')

    def validation_step(self, batch, batch_idx):
        """
        Implements a validation step.
        """
        # Compute the losses
        total_loss = self.compute_losses(batch, train_or_val='val')

        # The rest of the function computes the metrics for each task
        past_variables, past_datacubes, future_variables, future_datacubes = batch
        predictions = self.forward(past_variables, past_datacubes)

        # Before computing the metrics, we'll denormalize the future values, so that metrics are
        # computed in the original scale. The constants are stored in the dataset object.
        future_variables = self.val_dataset.denormalize_tabular_target(future_variables)

        # Compute the metrics for each task
        for task, task_params in self.tabular_tasks.items():
            # Denormalize the predictions using the task-specific denormalization function
            # so that the metrics are computed in the original scale
            predictions[task] = task_params['distrib_obj'].denormalize(predictions[task], task, self.train_dataset)
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
        # Denormalize the predictions using the task-specific denormalization function
        for task, task_params in self.tabular_tasks.items():
            predictions[task] = task_params['distrib_obj'].denormalize(predictions[task], task, self.train_dataset)
        return predictions
 
    def forward(self, past_variables, past_datacubes):
        """
        Parameters
        ----------
        past_variables: Mapping of str to tensor
            The past variables, with the keys being the names of the variables and the values
            being batches of shape (N, P).
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
        encoder_outputs = self.encoder(past_datacubes)  # Returns the skip connections
        # Apply the common linear model to the latent space and the past variables
        latent_space = encoder_outputs[-1]
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
            latent_space = latent_space.view(-1, *self.encoder.output_shape)
            datacube_preds = self.decoder(latent_space, encoder_outputs)
            # The target datacubes are concatenated along the channel dimension, retrieve them
            # individually
            start_channel = 0 
            for task, settings in self.datacube_tasks.items():
                end_channel = start_channel + settings['output_channels']
                predictions[task] = datacube_preds[:, start_channel:end_channel]
                start_channel = end_channel
        return predictions

    def configure_optimizers(self):
        """
        Configures the optimizer.
        """
        # If we are fine-tuning, freeze the encoder
        if self.training_cfg["freeze_encoder"]:
            print("/!\WARNING/!\ The encoder is frozen, only the decoder is trained.")
            for param in self.encoder.parameters():
                param.requires_grad = False
            # Set the encoder in eval mode so that the batchnorm layers are not updated
            self.encoder.eval()
        # If a single task is used, only update the parameters of the prediction head
        if self.training_cfg['trained_task'] is not None:
            print(f"/!\WARNING/!\ Only the parameters of the {self.training_cfg['trained_task']} "
                  "prediction head are updated.")
            # Freeze the parameters of the other prediction heads
            for task in self.tabular_tasks.keys():
                if task != self.training_cfg['trained_task']:
                    for param in self.prediction_heads[task].parameters():
                        param.requires_grad = False
        
        # Be careful to only update the parameters that require gradients
        updated_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(updated_params, lr=self.training_cfg['initial_lr'],
                                     weight_decay=self.training_cfg['weight_decay'])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=self.training_cfg['epochs'],
                                                                  eta_min=self.training_cfg['final_lr'])
        return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                }
