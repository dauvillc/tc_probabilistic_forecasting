"""
Tests the implementation of the quantile CRPS.
"""
import sys
sys.path.append("./")
import torch
from loss_functions.quantiles import QuantilesCRPS


if __name__ == "__main__":
    # Test 1 - Two quantiles are predicted only
    y = torch.tensor([2])
    pred = torch.tensor([[[1, 3]]])
    probas = torch.tensor([0.25, 0.75])
    crps = QuantilesCRPS(probas)
    assert abs(crps(pred, y) - torch.tensor(0.5417)).item() < 1e-3

    # Test 2 - Two quantiles, observation is lower than the lowest quantile
    y = torch.tensor([0])
    pred = torch.tensor([[[1, 3]]])
    probas = torch.tensor([0.25, 0.75])
    crps = QuantilesCRPS(probas)
    assert abs(crps(pred, y) - torch.tensor(3.041)).item() < 1e-3

    # Test 3 - Two quantiles, observation is higher than the highest quantile
    y = torch.tensor([4])
    pred = torch.tensor([[[1, 3]]])
    probas = torch.tensor([0.25, 0.75])
    crps = QuantilesCRPS(probas)
    assert abs(crps(pred, y) - torch.tensor(0.7917)).item() < 1e-3

    # Test 4 - Two quantiles, observation is exactly the second quantile
    y = torch.tensor([3])
    pred = torch.tensor([[[1, 3]]])
    probas = torch.tensor([0.25, 0.75])
    crps = QuantilesCRPS(probas)
    assert abs(crps(pred, y) - torch.tensor(0.5417)).item() < 1e-3

    # Test 5 - Three quantiles, observation is at the middle between
    # the first and second quantiles
    y = torch.tensor([1.5])
    pred = torch.tensor([[[1, 2, 3.5]]])
    probas = torch.tensor([0.25, 0.5, 0.75])
    crps = QuantilesCRPS(probas)
    assert abs(crps(pred, y) - torch.tensor(0.8802)).item() < 1e-3

