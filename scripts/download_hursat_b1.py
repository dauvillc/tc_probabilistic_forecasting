"""
Downloads the HURSAT-B1 dataset from the NOAA website.
"""
import sys
sys.path.append('./')
import os
import yaml
import urllib.request
import argparse
from bs4 import BeautifulSoup
from tqdm import tqdm


# URL to download the HURSAT-B1 dataset from
_HURSAT_B1_URL = 'https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/'


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Download the HURSAT-B1 dataset from the NOAA website.')
    parser.add_argument('-s', '--start_year', type=int, default=2000, help='The year to start downloading from.')
    parser.add_argument('-e', '--end_year', type=int, default=2015, help='The year to stop downloading at.')
    args = parser.parse_args()
    # Extract the start and end years
    start_year, end_year = args.start_year + 2000, args.end_year + 2000

    # Load the hursat-b1 paths from the config file
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    hursat_b1_path = config['paths']['hursat_b1']
    hursat_b1_list_path = config['paths']['hursat_b1_list']

    # For every year in the range, for every storm in the year, download the hursat-b1 data
    # and save it to the appropriate directory.
    for year in range(start_year, end_year + 1):
        # Create the directory for the year if it doesn't exist
        if not os.path.exists(os.path.join(hursat_b1_path, str(year))):
            os.makedirs(os.path.join(hursat_b1_path, str(year)))
        # Retrieve the list of all storms listed on the HURSAT webpage for that year
        # To do so, retrieve the HTML from the webpage, then parse all <td> tags,
        # and find those whose text begins with 'HURSAT_':
        with urllib.request.urlopen(_HURSAT_B1_URL + str(year)) as response:
            html = response.read()
        soup = BeautifulSoup(html, 'html.parser')
        storm_links = soup.find_all('td', string=lambda s: s.startswith('HURSAT_'))
        
        # For each storm, download the data and save it to the directory for that year.
        for storm_link in tqdm(storm_links):
            # Build the download url from the storm link.
            # The links are of the form
            # HURSAT_b1_v06_SID_NAME_c20170721.tar.gz
            storm_url = _HURSAT_B1_URL + str(year) + '/' + storm_link.string
            storm_id = storm_link.string.split('_')[3]
            # Check if there is already a directory for the storm.
            # If there isn't, create it, download the data, and extract it.
            storm_dir = os.path.join(hursat_b1_path, str(year), storm_id)
            if not os.path.exists(storm_dir):
                os.makedirs(storm_dir)
                urllib.request.urlretrieve(storm_url, os.path.join(storm_dir, storm_link.string))
                os.system('tar -xzf ' + os.path.join(storm_dir, storm_link.string) + ' -C ' + storm_dir)
                # Remove the tar.gz file
                os.remove(os.path.join(storm_dir, storm_link.string))
