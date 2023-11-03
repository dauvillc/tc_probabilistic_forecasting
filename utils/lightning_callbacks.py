"""
Implements callbacks for pytorch_lightning pipelines.
"""
from pytorch_lightning.callbacks import Callback


class MetricTracker(Callback):
    """
    Implements a callback to track metrics during training.
    Thanks @ayandas
    https://stackoverflow.com/questions/69276961/how-to-extract-loss-and-accuracy-from-logger-by-each-epoch-in-pytorch-lightning

    Parameters
    ----------
    batch_size : int
        The batch size used during training and validation.
    """
    def __init__(self, batch_size):
        self.train_loss = []
        self.val_loss = []
        self.batch_size = batch_size

    def on_train_epoch_end(self, trainer, pl_module):
        # Pytorch Lightning divides by the number of batches in one epoch, but not by
        # the batch size. We need to divide by the batch size to get the average loss
        # per sample
        self.train_loss.append(trainer.callback_metrics['train_loss'].item() / self.batch_size)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_loss.append(trainer.callback_metrics['val_loss'].item() / self.batch_size)
