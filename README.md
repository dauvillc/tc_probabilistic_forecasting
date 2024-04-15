# Comparing different inputs for tropical cyclones probabilistic intensity forecasting.
## Reproducibility
The pipeline is currently built to work with the [TCIR](https://www.csie.ntu.edu.tw/~htlin/program/TCIR/) dataset, as well as [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview).  
Futhermore, it relies on [Weights and Biases](https://wandb.ai/site) for logging training metrics and model checkpoints.  

[b]Note: All scripts should be run from the project's root directory.[/b]
### Dataset
This repo is radily evolving, and preparing the data for preprocessing will be made easier in the future.  
The instructions to currently reproduce the preprocessing are as follows:  
* Download the three h5 files from the TCIR webpage and extract them into any location; indicate the absolute path to that location in ```config.yml```.
* Download the ERA5 surface files (choosing any fields of interest), for the years 2003 to 2017 included, and store them in any location.
* Rename the ERA5 files to the format "YYYY_MM_surface.nc", and indicate in ```config.yml```, in paths/era5, the directory in which the ERA5 data is stored.
* Also indicate a directory where the ERA5 TC-centered patches will be saved, in ```config.yml/paths/era5_patches```.
* Run ```scipts/match_tcir_era5.py --rescale 0.07 -y Y``` to exract the ERA5 patches around the TCs for the year (2000 + Y).
* Indicate in ```config.yml/paths/tcir_preprocessed_dir``` the directory in which the preprocessed dataset should be stored.
* Run ```scripts/preprocess_tcir.py``` to produce the preprocessed dataset (which includes making the train / val / test splitting). Note that this will require a large amount of RAM (probably at least 64GB). The process uses Dask Arrays via xarray, and reducing the chunksize in ```preprocess_tcir.py``` will reduce the amount of required memory.
### Experiment
* Indicate in ```training_cfg.yml``` the settings of your experiment. ```name``` and ```group``` are relative to Weights and Biases. The ```input_channels``` parameter may be any field from either TCIR ('IR' or 'PMW'), or from ERA5 (e.g. 'u10', 'v10', 't2m' or 'msl').  
Unless making a probabilistic model or to evaluate on other tasks (MSLP, R35, LAT/LON, SSHS, RI), do not change the "tasks" section.
* Run ```experiments/training.py``` to train the model. This will require you to be logged in to W&B.
* Run ```experiments/make_predictions.py -i ID``` to make predictions on the test set using the model checkpoint uploaded to W&B during training. ID should be the Weights and Biases run id. The predictions and associated targets are saved in ```root_dir/data/predictions/ID``` where root_dir is the project's root directory.
* Run ```experiments/eval_crossval.py -i ID1 ID2 ... IDn -m MAE -n NAME``` to evalaute the predictions of models ```ID1, ..., IDn```. ```NAME``` is a name given to the evaluation and can be anything containing non-special characters. In particular, it is not the name of any W&B experiment. The results and figures are saved in ```results/NAME```.

