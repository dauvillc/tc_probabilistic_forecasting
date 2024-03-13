"""
Implements metrics whose computation is common to all distributions.
"""

import torch
from utils.utils import average_score, _SSHS_THRESHOLDS_


class SSHSBrierScore:
    """
    Evaluates the Brier score for a probabilistic forecast
    over the binary task "the intensity is at least of category C"
    where C is given.

    Parameters
    ----------
    cdf_fn: callable
        Function F such that F(predicted_params, y) is the CDF of the
        distribution with parameters predicted_params, evaluated at y.
        F must return a tensor of the same shape as y.
    cat: int between -1 and 5 (inclusive)
        The SSHS category to consider.
    """

    def __init__(self, cdf_fn, cat):
        self.cdf_fn = cdf_fn
        self.cat = cat
        self.threshold = torch.tensor([_SSHS_THRESHOLDS_[cat + 1]])

    def __call__(self, pred, y, reduce_mean="all"):
        """
        Parameters
        ----------
        pred: torch.Tensor
            Predicted parameters for the distribution.
        y: torch.Tensor of shape (N, T)
            Observations.
        reduce_mean: str
            Over which dimension(s) to average the Brier score.
            Can be "all", "samples", "time", "none".
        """
        # Move the threshold to the right device
        self.threshold = self.threshold.to(pred.device)
        # Compute P(Y <= cat)
        p = self.cdf_fn(pred, self.threshold.repeat(y.shape).to(pred.device))
        # Deduce P(Y > cat)
        p = 1 - p
        # Compute the binary target
        o = (y >= self.threshold).float()
        # Deduce the Brier score
        brier = (p - o) ** 2
        return average_score(brier, reduce_mean)


class CoveredCrps:
    """
    Evaluates the CRPS only on observations that are covered by the
    predicted distribution.
    Given that a sample has T time steps, its coverage is not necessarily
    0 or 1, but a value in [0, 1] that represents the proportion of time
    steps that are covered by the predicted distribution.
    Thus, a sample is considered covered if its coverage is above a given
    threshold.

    Parameters
    ----------
    crps_fn: callable
        A function CRPS(pred, y, reduce_mean) that computes the CRPS.
    coverage_fn: callable
        A function coverage(pred, y, reduce_mean) that computes the coverage.
    coverage_threshold: float
        The threshold above which a time step is considered covered.
    """

    def __init__(self, crps_fn, coverage_fn, coverage_threshold=1):
        self.crps_fn = crps_fn
        self.coverage_fn = coverage_fn
        self.coverage_threshold = coverage_threshold

    def __call__(self, pred, y, reduce_mean="all"):
        """
        Parameters
        ----------
        pred: torch.Tensor
            Predicted distribution.
        y: torch.Tensor
            Target values.
        reduce_mean: str
            Over which dimension(s) to average the CRPS.
            Can be "all", "samples", "time", "none".
        """
        coverage = self.coverage_fn(pred, y, reduce_mean="none")
        crps = self.crps_fn(pred, y, reduce_mean="none")
        crps = crps[coverage >= self.coverage_threshold]
        return average_score(crps, reduce_mean)
