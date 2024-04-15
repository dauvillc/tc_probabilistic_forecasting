# Comparing different inputs for tropical cyclones probabilistic intensity forecasting.
## Reproducibility
The pipeline is currently built to work with the [TCIR](https://www.csie.ntu.edu.tw/~htlin/program/TCIR/) dataset, as well as [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview).  
Futhermore, it relies on [Weights and Biases](https://wandb.ai/site) for logging training metrics and model checkpoints.  
This repo is radily evolving, and preparing the data for preprocessing will be made easier in the future.  
The instructions to currently reproduce the preprocessing are as follows:  
* Download the three h5 files from the TCIR webpage and extract them into any location; indicate the absolute path to that location in ```config.yml```.
* Download the ERA5 surface files (choosing any fields of interest), for the years 2003 to 2017 included, and store them in any location.
* Rename the ERA5 files to the format "YYYY_MM_surface.nc", and indicate in ```config.yml```, in paths/era5, the directory in which the ERA5 data is stored.
* Also indicate a directory where the ERA5 TC-centered patches will be saved, in ```config.yml/paths/era5_patches```.
* Run ```scipts/match_tcir_era5.py --rescale 0.07 -y Y``` to exract the ERA5 patches around the TCs for the year (2000 + Y).
* Indicate in ```config.yml/paths/tcir_preprocessed_dir``` the directory in which the preprocessed dataset should be stored.
* Run ```scripts/preprocess_tcir.py``` to produce the preprocessed dataset (which includes making the train / val / test splitting). Note that this will require a large amount of RAM (probably at least 64GB). The process uses Dask Arrays via xarray, and reducing the chunksize in ```preprocess_tcir.py``` will reduce the amount of required memory.
