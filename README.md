# Comparing different inputs for tropical cyclones probabilistic intensity forecasting.
## Reproducibility
The pipeline is currently built to work with the [TCIR](https://www.csie.ntu.edu.tw/~htlin/program/TCIR/) dataset, as well as [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview).  
Futhermore, it relies on [Weights and Biases](https://wandb.ai/site) for logging training metrics and model checkpoints.  
This repo is radily evolving, and preparing the data for preprocessing will be made easier in the future.  
The instructions to currently reproduce the preprocessing are as follows:  
* Download the three h5 files from the TCIR webpage and extract them into any location; indicate the absolute path to that location in ```config.yml```.

