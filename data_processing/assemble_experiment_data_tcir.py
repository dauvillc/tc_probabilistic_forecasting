"""
Implements a function to assemble the dataset for the experiment.
"""
import torch
from data_processing.custom_dataset_v2 import SuccessiveStepsDataset
from data_processing.datasets import load_tcir
from utils.datacube import datacube_to_tensor
from utils.train_test_split import train_val_test_split
from utils.utils import hours_to_sincos



def load_dataset(cfg, input_variables, tasks):
    """
    Assembles the dataset, performs the train/val/test split and creates the
    datasets and data loaders.

    Parameters
    ----------
    cfg: mapping of str to Any
        The configuration of the experiment.
    input_variables : list of str
        The list of the input variables.
    tasks: Mapping of str to Mapping
        The tasks to perform. The keys are the task names, and the values are
        mappings containing the task parameters, including:
        - 'output_variables': list of str
            The list of the output variables.

    Returns
    -------
    train_dataset : torch.utils.data.Dataset
    val_dataset : torch.utils.data.Dataset
    train_loader : torch.utils.data.DataLoader
    val_loader : torch.utils.data.DataLoader
    """
    past_steps, future_steps = cfg['experiment']['past_steps'], cfg['experiment']['future_steps']
    # ====== LOAD DATASET ====== #
    # Load the TCIR dataset
    tcir_info, tcir_datacube = load_tcir()
    print('TCIR dataset loaded')
    print('Memory usage: {:.2f} GB'.format(tcir_datacube.nbytes / 1e9))

    # Add a column with the sin/cos encoding of the hours, which will be used as input
    # to the model
    sincos_hours = hours_to_sincos(tcir_info['ISO_TIME'])
    tcir_info['HOUR_SIN'], tcir_info['HOUR_COS'] = sincos_hours[:, 0], sincos_hours[:, 1] 

    # Convert the datacube to a tensor
    tcir_datacube = datacube_to_tensor(tcir_datacube)

    # ====== TRAIN/VAL/TEST SPLIT ====== #
    # Split the dataset into train, validation and test sets
    train_index, val_index, test_index = train_val_test_split(tcir_info,
                                                              train_size=0.6,
                                                              val_size=0.2,
                                                              test_size=0.2)
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
    # Create the train and validation datasets.
    train_dataset = SuccessiveStepsDataset(train_trajs, input_variables, tasks,
                                           {'tcir': train_patches}, ['tcir'], [],
                                           past_steps, future_steps)
    val_dataset = SuccessiveStepsDataset(val_trajs, input_variables, tasks,
                                         {'tcir': val_patches}, ['tcir'], [],
                                         past_steps, future_steps)
    # Normalize the data. For the validation dataset, we use the mean and std of the training dataset.
    # The normalization constants are saved in the tasks dictionary.
    train_dataset.normalize_inputs()
    train_dataset.normalize_outputs(save_statistics=True)
    val_dataset.normalize_inputs(other_dataset=train_dataset)
    # Create the train and validation data loaders
    batch_size = cfg['training_settings']['batch_size']
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataset, val_dataset, train_loader, val_loader
