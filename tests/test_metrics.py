"""
Tests the data loading and evaluation functions.
"""
import sys
sys.path.append("./")
import argparse
import yaml
import numpy as np
from tasks.intensity import intensity_dataset, plot_intensity_bias


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--prediction_steps", type=int, default=1,
                        help="Number of time steps to predict.")
    args = parser.parse_args()
    predicted_steps = args.prediction_steps
    # Load the configuration file
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Load the trajectory forecasting dataset
    all_trajs = intensity_dataset()

    # Create a fake ground truth, in which the intensity is constant over all
    # time steps, and its value is taken from all_trajs
    y_true = np.tile(all_trajs['INTENSITY'].values, (predicted_steps, 1)).T
    # Simulate a model that predicts the same intensity for all time steps, plus
    # Gaussian noise whose variance increases at each time step.
    y_pred = y_true.copy()
    for i in range(predicted_steps):
        y_pred[:, i] += np.random.normal(loc=i*0.2, scale=i + 1, size=y_pred.shape[0])

    # Plot the bias
    plot_intensity_bias(y_true, y_pred, savepath="figures/tests/intensity_bias.png")
