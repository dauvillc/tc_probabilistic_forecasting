"""
Implements a function to assemble the dataset for the experiment.
"""

import torch
from data_processing.custom_dataset import SuccessiveStepsDataset
from data_processing.datasets import load_tcir
from utils.sampling import inverse_intensity_sampler
from utils.datacube import datacube_to_tensor


def load_dataset(cfg, input_variables, tabular_tasks, subset):
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
    subset : str
        'train', 'val' or 'test'.

    Returns
    -------
    dataset : torch.utils.data.Dataset
    loader : torch.utils.data.DataLoader
    """
    # ====== LOAD DATASET ====== #
    # Load the TCIR dataset
    tcir_info, tcir_datacube, info_means, info_stds, datacube_means, datacube_stds = load_tcir(
        subset, channels=cfg['experiment']['input_channels']
    )
    print(f"TCIR {subset} dataset loaded")
    print("Memory usage: {:.2f} GB".format(tcir_datacube.nbytes / 1e9))

    # Convert the datacube to a tensor
    tcir_datacube = datacube_to_tensor(tcir_datacube)

    # ====== DATASET CREATION ====== #
    # Create the custom dataset
    dataset = SuccessiveStepsDataset(
        subset,
        tcir_info,
        input_variables,
        tabular_tasks,
        {"tcir": tcir_datacube},
        info_means,
        info_stds,
        datacube_means,
        datacube_stds,
        cfg,
    )
    # Create the sampler and data loaders
    batch_size = cfg["training_settings"]["batch_size"]
    num_workers = (
        cfg["training_settings"]["num_workers"] if "num_workers" in cfg["training_settings"] else 0
    )
    persistent_workers = num_workers > 0
    if subset == "train":
        if ("sampling_weights" in cfg["training_settings"]) and (
            cfg["training_settings"]["sampling_weights"]
        ):
            # Create the sampler
            sampler = inverse_intensity_sampler(
                dataset.get_sample_intensities(), plot_weights="figures/weights.png"
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
            )
        else:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
            )
    else:
        # Validation and test sets - no data aug, no shuffling
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    return dataset, loader
