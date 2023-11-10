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
    batch_size : int, optional
        The batch size used during training and validation.
        If specified, the loss will be divided by the batch size.
    """
    def __init__(self, batch_size=None):
        self.train_loss = []
        self.val_loss = []
        self.batch_size = batch_size

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics['train_loss'].item()
        if self.batch_size is not None:
            loss /= self.batch_size
        self.train_loss.append(loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics['val_loss'].item()
        if self.batch_size is not None:
            loss /= self.batch_size
        self.val_loss.append(loss)
