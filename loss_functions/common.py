"""
Implements metrics whose computation is common to all distributions.
"""


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

    def __call__(self, pred, y, reduce_mean=True):
        """
        Parameters
        ----------
        pred: torch.Tensor
            Predicted distribution.
        y: torch.Tensor
            Target values.
        reduce_mean: bool
            Whether to average the result over the batch dimension.
        """
        coverage = self.coverage_fn(pred, y, reduce_mean=False)
        crps = self.crps_fn(pred, y, reduce_mean=False)
        crps = crps[coverage >= self.coverage_threshold]
        if reduce_mean:
            return crps.mean()
        return crps
