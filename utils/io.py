"""
Implements functions to manage data saving and loading.
"""
import os
import torch
from pathlib import Path


_PREDS_DIR_ = Path('data') / 'predictions'


def write_tensors_dict(tensors_dict, save_dir):
    """
    Saves a dictionary of tensors to a directory.
    The keys must be strings and the values must be torch.Tensors. For each
    pair (K, V), V is saved under save_dir/K.pt.

    Parameters
    ----------
    tensors_dict : dict str -> torch.Tensor
        Dictionary containing the tensors to save.
    save_dir: str
        Path to the directory in which the dictionary will be saved. Will be
        created if necessary.
    """
    os.makedirs(save_dir, exist_ok=True)
    for key, tensor in tensors_dict.items():
        print("Saving key ", key, " to ", save_dir)
        torch.save(tensor, os.path.join(save_dir, key + ".pt"))


def load_tensors_dict(load_dir):
    """
    Loads a dictionary of tensors from a directory.

    Parameters
    ----------
    load_dir: str
        Path to the directory from which the dictionary will be loaded.
    """
    device = torch.device("cpu")
    tensors_dict = {}
    for file in os.listdir(load_dir):
        if file.endswith(".pt"):
            print("Loading ", file, " from ", load_dir)
            key = file[:-3]
            tensor = torch.load(os.path.join(load_dir, file), device)
            tensors_dict[key] = tensor
    return tensors_dict


def load_predictions_and_targets(run_ids):
    """
    Loads the predictions and corresponding targets of multiple models at once.
    The predictions must have been already computed and saved
    via make_predictions.py .

    Parameters
    ----------
    run_ids: list of str
        List of the run ids of the models whose predictions are to be loaded.
    
    Returns
    -------
    predictions: Mapping str -> (Mapping str -> torch.Tensor)
        Mapping from run id to a dictionary of predictions, which maps
        a task name to a tensor of predictions.
    targets: Mapping str -> (Mapping str -> torch.Tensor)
        Target values for each task, in the same format as predictions.
    """
    predictions, targets = {}, {}
    for run_id in run_ids:
        predictions[run_id] = load_tensors_dict(_PREDS_DIR_ / run_id / "predictions" / "final")
        targets[run_id] = load_tensors_dict(_PREDS_DIR_ / run_id / "targets" / "final")
    return predictions, targets
