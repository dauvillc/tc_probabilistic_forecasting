"""
Implements the training and evaluation tools using the Lightning framework.
"""

import torch
from torch import nn
import pytorch_lightning as pl
from models.spatial_encoder import SpatialEncoder
from models.temporal_encoder import TemporalEncoder
from models.linear import PredictionHead, MultivariatePredictionHead
from utils.predictions import ResidualPrediction


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
    tabular_tasks: Mapping of str to Mapping
        The tasks whose targets are vectors, with the keys being the task names and the values
        being the task parameters, including:
        - 'output_variables': list of str
            The names of the output variables for the task.
        - 'output_size': int
            The number of values to predict for the task.
        - 'distrib_obj': Distribution object implementing loss_function, metrics
            and optionally activation.
    dataset: SuccessiveStepsDataset
        Pointer to the dataset object, used to denormalize the predictions and targets.
    cfg: dict
        The configuration dictionary.
    """

    def __init__(
        self,
        input_datacube_shape,
        tabular_tasks,
        dataset,
        cfg,
    ):
        super().__init__()
        self.tabular_tasks = tabular_tasks
        self.dataset = dataset
        self.model_cfg = cfg["model_hyperparameters"]
        self.training_cfg = cfg["training_settings"]
        self.input_datacube_shape = input_datacube_shape
        past_steps = cfg["experiment"]["past_steps"]
        self.target_steps = cfg["experiment"]["target_steps"]
        self.patch_size = cfg["experiment"]["patch_size"]
        C, P, H, W = input_datacube_shape

        # Create the spatial encoder
        self.spatial_encoder = SpatialEncoder(
            C,
            self.model_cfg["spatial_encoder"]["n_blocks"],
            self.model_cfg["spatial_encoder"]["base_channels"],
            self.model_cfg["spatial_encoder"]["base_block"],
            kernel_size=3,
        )
        C, H, W = self.spatial_encoder.output_size((C, H, W))
        # Create the temporal encoder
        C_out = C // self.model_cfg["temporal_encoder"]["reduction_factor"]
        self.temporal_encoder = TemporalEncoder(
            C, H, W, C_out, 3, 7, self.model_cfg["temporal_encoder"]["n_blocks"]
        )
        latent_size = C_out * past_steps * H * W
        context_size = self.dataset.context_size()

        # Create a prediction head which will predict the location of the distribution of Y_0
        # for each task
        self.location_head = nn.ModuleDict({})
        head_reduction_factor = self.model_cfg["prediction_head"]["reduction_factor"]
        for task, task_params in tabular_tasks.items():
            self.location_head[task] = PredictionHead(
                latent_size, context_size, 1, 1, head_reduction_factor
            )

        # Create the residual heads which will predict the distribution of the residual
        # P(Y_t - Y_0 | Y_0) for each task
        self.residual_heads = nn.ModuleDict({})
        for task, task_params in tabular_tasks.items():
            # Create the prediction head. If the task is to predict a distribution at each
            # time step, use PredictionHead. Otherwise, use MultivariatePredictionHead.
            if task_params["distrib_obj"].is_multivariate:
                self.residual_heads[task] = MultivariatePredictionHead(
                    latent_size, task_params["output_size"]
                )
            else:
                self.residual_heads[task] = PredictionHead(
                    latent_size,
                    context_size,
                    task_params["output_size"],
                    len(self.target_steps),
                    head_reduction_factor,
                )

    def compute_losses(self, batch, train_or_val="train"):
        """
        Computes the losses for a single batch (subfunction of training_step
        and validation_step).
        """
        past_variables, past_datacubes, target_locations, target_residuals = batch
        predictions = self.forward(past_variables, past_datacubes)
        reduce_mean = "all"
        # Compute the indivual losses for each task
        losses = {}
        for task, task_params in self.tabular_tasks.items():
            # Loss for the location prediction (predict Y_0)
            location_loss = predictions.loc_distrib.loss_function(
                predictions.locations[task], target_locations[task], reduce_mean=reduce_mean
            )
            # Log the loss
            self.log(
                f"{train_or_val}_location_loss_{task}",
                location_loss,
                on_step=False,
                on_epoch=True,
            )
            # Loss for the residual prediction (predict Y_t - Y_0)
            residual_loss = predictions.distribs[task].loss_function(
                predictions.residuals[task], target_residuals[task], reduce_mean=reduce_mean
            )
            self.log(
                f"{train_or_val}_residual_loss_{task}",
                residual_loss,
                on_step=False,
                on_epoch=True,
            )
            # Sum the losses to get the total loss for the task
            losses[task] = location_loss + residual_loss
        # Compute the total loss
        # In the case of single-task finetuning, the total loss is the loss of the trained task
        if self.training_cfg["trained_task"] is not None:
            total_loss = losses[self.training_cfg["trained_task"]]
        else:
            total_loss = sum(losses.values())
        self.log(
            f"{train_or_val}_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return total_loss

    def training_step(self, batch, batch_idx):
        """
        Implements a training step.
        """
        return self.compute_losses(batch, train_or_val="train")

    def validation_step(self, batch, batch_idx):
        """
        Implements a validation step.
        """
        # Compute the losses
        total_loss = self.compute_losses(batch, train_or_val="val")

        # The rest of the function computes the metrics for each task
        past_variables, past_datacubes, true_locations, true_residuals = batch
        predictions = self.forward(past_variables, past_datacubes)

        # Before computing the metrics, we'll denormalize the targets and predictions, so that metrics are
        # computed in the original scale. The constants are stored in the dataset object.
        true_locations = self.dataset.denormalize_tabular_target(true_locations)
        true_residuals = self.dataset.denormalize_tabular_target(true_residuals, residuals=True)
        # Denormalize the predictions
        predictions = predictions.denormalize(self.dataset)

        # Compute the metrics for each task
        for task, task_params in self.tabular_tasks.items():
            # Compute the distribution of the final predictions (Location + Residual)
            task_preds = predictions.final_predictions(task)
            # Compute the final targets (Location + Residual)
            targets = true_locations[task] + true_residuals[task]
            for metric_name, metric in task_params["distrib_obj"].metrics.items():
                # Compute the metric in the original scale
                metric_value = metric(task_preds, targets)
                self.log(
                    f"val_{metric_name}_{task}",
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                )

        return total_loss

    def predict_step(self, batch, batch_idx=0, dataloader_idx=None):
        """
        Implements a prediction step.
        """
        past_variables, past_datacubes, true_locations, true_residuals = batch
        predictions = self.forward(past_variables, past_datacubes)
        # Denormalize the predictions
        predictions = predictions.denormalize(self.dataset)
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
        ResidualPrediction object
            Object which contains the predictions (location, residual distribution, and complete
            distribution) for each task.
        """
        # Concatenate the past datacubes into a single tensor along the channel dimension
        past_datacubes = torch.cat(list(past_datacubes.values()), dim=1)
        # Concatenate the context variables into a single tensor
        past_variables = torch.cat(list(past_variables.values()), dim=1)
        # Apply the spatial encoder
        latent_space = self.spatial_encoder(past_datacubes)
        # Apply the temporal encoder
        latent_space = self.temporal_encoder(latent_space)
        # Flatten the latent space
        latent_space = latent_space.view(latent_space.size(0), -1)
        # Apply the prediction heads
        predictions = ResidualPrediction()
        for task, task_params in self.tabular_tasks.items():
            # Predict the location of the distribution
            location = self.location_head[task](latent_space, past_variables)
            # Predict the residual distribution
            residuals = self.residual_heads[task](latent_space, past_variables)
            # Store the predictions in the ResidualPrediction object, which also
            # calls the activation function of the distribution object
            predictions.add(task, location, residuals, task_params["distrib_obj"])
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
        if self.training_cfg["trained_task"] is not None:
            print(
                f"/!\WARNING/!\ Only the parameters of the {self.training_cfg['trained_task']} "
                "prediction head are updated."
            )
            # Freeze the parameters of the other prediction heads
            for task in self.tabular_tasks.keys():
                if task != self.training_cfg["trained_task"]:
                    for param in self.residual_heads[task].parameters():
                        param.requires_grad = False

        # Be careful to only update the parameters that require gradients
        updated_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            updated_params,
            lr=self.training_cfg["initial_lr"],
            betas=(self.training_cfg["beta1"], self.training_cfg["beta2"]),
            weight_decay=self.training_cfg["weight_decay"],
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.training_cfg["epochs"],
            eta_min=self.training_cfg["final_lr"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
