"""
Implements functions to manage data saving and loading.
"""
import os
import torch


_PREDS_DIR_ = os.path.join('data', 'predictions')
_TARGETS_DIR_ = os.path.join('data', 'targets')


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
    tensors_dict = {}
    for file in os.listdir(load_dir):
        if file.endswith(".pt"):
            print("Loading ", file, " from ", load_dir)
            key = file[:-3]
            tensor = torch.load(os.path.join(load_dir, file))
            tensors_dict[key] = tensor
    return tensors_dict


def load_predictions(run_ids):
    """
    Loads the predictions of multiple models at once.
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
    """
    predictions = {}
    for run_id in run_ids:
        predictions[run_id] = load_tensors_dict(os.path.join(_PREDS_DIR_, run_id))
    return predictions


def load_targets():
    """
    Loads the targets of all tasks.

    Returns
    -------
    targets: Mapping str -> torch.Tensor
        Mapping from task name to the tensor of targets.
    """
    return load_tensors_dict(_TARGETS_DIR_)
