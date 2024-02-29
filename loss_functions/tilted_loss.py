"""
Defines the TiltedLoss class, which applies exponential tilting to a
given loss function.
"""
import torch

class TiltedLoss(torch.nn.Module):
    """
    Applies exponential tilting to a set of given loss values.
    Based on TERM by Tian et al. (2023) "On Tilted Losses in Machine Learning"
    http://arxiv.org/abs/2109.06141

    Parameters
    ----------
    t: float
        The tilt parameter. t=0 doesn't alter the loss.
        t > 0 gives more weight to samples with larger loss.
        t < 0 gives more weight to samples with smaller loss.
    """
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, loss):
        """
        Applies exponential tilting to the given loss values.

        Parameters
        ----------
        loss: torch.Tensor of shape (N,)
            The loss values to be tilted.

        Returns
        -------
        torch.Tensor of shape (N,)
            The tilted loss values.
        """
        tilted_values = torch.exp(self.t * loss)
        log_mean = torch.log(torch.mean(tilted_values))
        return log_mean / self.t

    def __call__(self, loss):
        """
        Applies exponential tilting to the given loss values.

        Parameters
        ----------
        loss: torch.Tensor of shape (N,)
            The loss values to be tilted.

        Returns
        -------
        torch.Tensor of shape (N,)
            The tilted loss values.
        """
        return self.forward(loss)
