"""
Implements a function to assemble the dataset for the experiment.
"""
import torch
from tasks.intensity import intensity_dataset
from data_processing.formats import SuccessiveStepsDataset, datacube_to_tensor
from data_processing.datasets import load_hursat_b1, load_era5_patches
from utils.train_test_split import train_val_test_split
from utils.utils import hours_to_sincos



def load_dataset(args, input_variables, output_variables):
    """
    Assembles the dataset, performs the train/val/test split and creates the
    datasets and data loaders.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments of the script.
    input_variables : list of str
        The list of the input variables.
    output_variables : list of str
        The list of the variables to predict.

    Returns
    -------
    train_dataset : torch.utils.data.Dataset
    val_dataset : torch.utils.data.Dataset
    train_loader : torch.utils.data.DataLoader
    val_loader : torch.utils.data.DataLoader
    """
    past_steps, future_steps = args.past_steps, args.future_steps
    # Load the trajectory forecasting dataset
    all_trajs = intensity_dataset()
    # Add a column with the sin/cos encoding of the hours, which will be used as input
    # to the model
    sincos_hours = hours_to_sincos(all_trajs['ISO_TIME'])
    all_trajs['HOUR_SIN'], all_trajs['HOUR_COS'] = sincos_hours[:, 0], sincos_hours[:, 1] 

    # Load the HURSAT-B1 data associated to the dataset
    # We need to load the hursat data even if we don't use it, because we need to
    # keep only the storms for which we have HURSAT-B1 data to fairly compare the
    # runs.
    found_storms, hursat_data = load_hursat_b1(all_trajs, use_cache=True, verbose=True)
    # Keep only the storms for which we have HURSAT-B1 data
    all_trajs = all_trajs.merge(found_storms, on=['SID', 'ISO_TIME'])
    # Load the right patches depending on the input data
    if args.input_data == "era5":
        # Load the ERA5 patches associated to the dataset
        atmo_patches, surface_patches = load_era5_patches(all_trajs, load_atmo=False)
        patches = datacube_to_tensor(surface_patches)
    elif args.input_data == "hursat":
        patches = datacube_to_tensor(hursat_data)
    elif args.input_data == "era5+hursat":
        # Load the ERA5 patches associated to the dataset
        atmo_patches, surface_patches = load_era5_patches(all_trajs, load_atmo=False)
        era5_patches = datacube_to_tensor(surface_patches)
        hursat_patches = datacube_to_tensor(hursat_data)
        # Concatenate the patches along the channel dimension
        patches = torch.cat([era5_patches, hursat_patches], dim=1)
    else:
        raise ValueError("The input data must be 'era5', 'hursat' or 'era5+hursat'.")


    # ====== TRAIN/VAL/TEST SPLIT ====== #
    # Split the dataset into train, validation and test sets
    train_index, val_index, test_index = train_val_test_split(all_trajs,
                                                              train_size=0.6,
                                                              val_size=0.2,
                                                              test_size=0.2)
    train_trajs = all_trajs.iloc[train_index]
    val_trajs = all_trajs.iloc[val_index]
    test_trajs = all_trajs.iloc[test_index]

    print(f"Number of trajectories in the training set: {len(train_trajs)}")
    print(f"Number of trajectories in the validation set: {len(val_trajs)}")
    print(f"Number of trajectories in the test set: {len(test_trajs)}")

    # ====== DATASET CREATION ====== #
    # Split the patches into train, validation and test sets
    train_patches = patches[train_index]
    val_patches = patches[val_index]

    # Create the train and validation datasets. For the validation dataset,
    # we need to normalise the data using the statistics from the train dataset.
    yield_input_variables = len(input_variables) > 0
    train_dataset = SuccessiveStepsDataset(train_trajs, train_patches, past_steps, future_steps,
                                           input_variables, output_variables,
                                           yield_input_variables=yield_input_variables)
    val_dataset = SuccessiveStepsDataset(val_trajs, val_patches, past_steps, future_steps,
                                         input_variables, output_variables,
                                         yield_input_variables=yield_input_variables,
                                         normalise_from=train_dataset)
    # Create the train and validation data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_dataset, val_dataset, train_loader, val_loader
