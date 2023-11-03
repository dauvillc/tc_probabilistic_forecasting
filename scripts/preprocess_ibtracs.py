"""
Clément Dauvilliers - 2020-10-12
Preprocesses the IBTrACS dataset to extract the relevant variables regarding
tropical storms trajectory and intensity prediction.
"""

import pandas as pd
import datetime as dt
import argparse
import yaml


# List of variables to keep from the dataset
_SELECTED_VARS_ = ['SID', 'NAME', 'ISO_TIME', 'LAT', 'LON', 'BASIN', 'TRACK_TYPE', 'USA_WIND']


if __name__ == "__main__":
    # Load the configuration file.
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    # Get the path to the IBTrACS dataset.
    ibtracs_path = config['paths']['ibtracs']
    output_path = config['paths']['ibtracs_preprocessed']
    # Arguments parsing
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--start_date', type=str, default='2000-01-01',
                           help='First date to consider for the dataset.')
    argparser.add_argument('--end_date', type=str, default='2021-12-31',
                           help='Last date to consider for the dataset.')
    args = argparser.parse_args()

    # Load the IBTrACS data and select the relevant variables.
    # Skip the first row after the header, as it contains the units.
    ibtracs_df = pd.read_csv(ibtracs_path, skiprows=[1], usecols=_SELECTED_VARS_, na_values=[' '],
                             parse_dates=['ISO_TIME'])
    # Filter out tracks that are provisional or side tracks.
    ibtracs_df = ibtracs_df[ibtracs_df['TRACK_TYPE'] == 'main']

    # =========== TEMPORAL SELECTION ===========
    # For each storm, retrieve the time of its first observation.
    first_obs = ibtracs_df.groupby('SID')['ISO_TIME'].min().reset_index()
    # Filter the storms to only keep those that have their first observation between the start and end dates.
    filtered_storms = first_obs[(first_obs['ISO_TIME'] >= dt.datetime.strptime(args.start_date, '%Y-%m-%d')) &
                                (first_obs['ISO_TIME'] <= dt.datetime.strptime(args.end_date, '%Y-%m-%d'))]
    # Filter the IBTrACS dataset to only keep the storms that have their first observation between the start and end
    # dates.
    ibtracs_df = ibtracs_df[ibtracs_df['SID'].isin(filtered_storms['SID'])]

    # Resample the dataset to have a frequency of 6 hours.
    ibtracs_df = ibtracs_df.set_index('ISO_TIME').groupby('SID').resample('6H').asfreq().drop(columns=['SID']).reset_index()

    # Find the storms for which at least one time step lacks the lat or lon information.
    missing_lat_lon = ibtracs_df[ibtracs_df[['LAT', 'LON']].isna().any(axis=1)]['SID'].unique()
    # Remove the storms that have at least one time step lacking the lat or lon information.
    ibtracs_df = ibtracs_df[~ibtracs_df['SID'].isin(missing_lat_lon)]

    # =========== UNIT CONVERSION ===========
    # Create a column for the Saffir-Simpson scale category of the storm.
    ibtracs_df['SS_SCALE'] = pd.cut(ibtracs_df['USA_WIND'],
                                    [-1, 34, 63, 82, 95, 112, 136, 250],
                                    labels=[-1, 0, 1, 2, 3, 4, 5],
                                    right=True)
    # Convert the wind speed from knots to m/s.
    ibtracs_df['USA_WIND'] = ibtracs_df['USA_WIND'] * 0.514444
    # Convert the longitude from [-180, 180) to [0, 350) for consistency with ERA5.
    ibtracs_df['LON'] = ibtracs_df['LON'].apply(lambda x: x + 360 if x < 0 else x)

    # Save the preprocessed dataset.
    ibtracs_df.to_csv(output_path, index=False)
