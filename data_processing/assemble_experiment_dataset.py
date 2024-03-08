"""
Implements a function to assemble the dataset for the experiment.
"""

import torch
from data_processing.custom_dataset import SuccessiveStepsDataset
from data_processing.datasets import load_tcir
from utils.sampling import inverse_intensity_sampler
from utils.datacube import datacube_to_tensor
from utils.train_test_split import train_val_test_split
from utils.utils import hours_to_sincos


def load_dataset(cfg, input_variables, tabular_tasks, datacube_tasks):
    """
    Assembles the dataset, performs the train/val/test split and creates the
    datasets and data loaders.

    Parameters
    ----------
    cfg: mapping of str to Any
        The configuration of the experiment.
    input_variables : list of str
        The list of the input variables.
    tabular_tasks: Mapping of str to Mapping
        The tasks whose targets are vectors. The keys are the task names, and the values are
        mappings containing the task parameters, including:
        - 'output_variables': list of str
            The list of the output variables.
    datacube_tasks: list of str
        The list of the output datacubes.

    Returns
    -------
    train_dataset : torch.utils.data.Dataset
    val_dataset : torch.utils.data.Dataset
    train_loader : torch.utils.data.DataLoader
    val_loader : torch.utils.data.DataLoader
    """
    # ====== LOAD DATASET ====== #
    # Load the TCIR dataset
    tcir_info, tcir_datacube = load_tcir()
    print("TCIR dataset loaded")
    print("Memory usage: {:.2f} GB".format(tcir_datacube.nbytes / 1e9))

    # Add a column with the sin/cos encoding of the hours, which will be used as input
    # to the model
    sincos_hours = hours_to_sincos(tcir_info["ISO_TIME"])
    tcir_info["HOUR_SIN"], tcir_info["HOUR_COS"] = sincos_hours[:, 0], sincos_hours[:, 1]

    # Convert the datacube to a tensor
    tcir_datacube = datacube_to_tensor(tcir_datacube)

    # ====== TRAIN/VAL/TEST SPLIT ====== #
    # Split the dataset into train, validation and test sets
    train_index, val_index, test_index = train_val_test_split(
        tcir_info, train_size=0.6, val_size=0.2, test_size=0.2
    )
    # Trajectory
    train_trajs = tcir_info.iloc[train_index]
    val_trajs = tcir_info.iloc[val_index]
    test_trajs = tcir_info.iloc[test_index]
    # Patches
    train_patches = tcir_datacube[train_index]
    val_patches = tcir_datacube[val_index]

    print(f"Number of trajectories in the training set: {len(train_trajs)}")
    print(f"Number of trajectories in the validation set: {len(val_trajs)}")
    print(f"Number of trajectories in the test set: {len(test_trajs)}")

    # ====== DATASET CREATION ====== #
    # Data augmentation is only applied to the training dataset if requested
    apply_data_aug = False
    if "data_augmentation" in cfg["training_settings"]:
        apply_data_aug = cfg["training_settings"]["data_augmentation"]
    # Create the train and validation datasets.
    train_dataset = SuccessiveStepsDataset(
        train_trajs,
        input_variables,
        tabular_tasks,
        {"tcir": train_patches},
        ["tcir"],
        datacube_tasks,
        cfg,
        random_rotations=apply_data_aug,
    )
    val_dataset = SuccessiveStepsDataset(
        val_trajs,
        input_variables,
        tabular_tasks,
        {"tcir": val_patches},
        ["tcir"],
        datacube_tasks,
        cfg,
        random_rotations=False,
    )
    # Normalize the data. For the validation dataset, we use the mean and std of the training dataset.
    # The normalization constants are saved in the tabular_tasks dictionary.
    train_dataset.normalize_inputs()
    train_dataset.normalize_outputs(save_statistics=True)
    val_dataset.normalize_inputs(other_dataset=train_dataset)
    val_dataset.normalize_outputs(other_dataset=train_dataset)
    # Create the sampler and data loaders
    batch_size = cfg["training_settings"]["batch_size"]
    num_workers = (
        cfg["training_settings"]["num_workers"] if "num_workers" in cfg["training_settings"] else 0
    )
    if ("sampling_weights" in cfg["training_settings"]) and (
        cfg["training_settings"]["sampling_weights"]
    ):
        # Create the sampler
        sampler = inverse_intensity_sampler(
            train_dataset.get_sample_intensities(), plot_weights="figures/weights.png"
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, train_loader, val_loader
