########################################################################
#
# Functions for downloading and re-sampling weather-data
# for 5 cities in Denmark between 1980-2018.
#
# The raw data was obtained from:
#
#   National Climatic Data Center (NCDC) in USA
#   https://www7.ncdc.noaa.gov/CDO/cdoselect.cmd
#
# Note that the NCDC's database functionality may change soon, and
# that the CSV-file needed some manual editing before it could be read.
# See the function _convert_raw_data() below for inspiration if you
# want to convert a new data-file from NCDC's database.
#
# Implemented in Python 3.6
#
# Usage:
# 1) Set the desired storage directory in the data_dir variable.
# 2) Call maybe_download_and_extract() to download the data-set
#    if it is not already located in the given data_dir.
# 3) Either call load_original_data() or load_resampled_data()
#    to load the original or resampled data for use in your program.
#
# Format:
# The raw data-file from NCDC is not included in the downloaded archive,
# which instead contains a cleaned-up version of the raw data-file
# referred to as the "original data". This data has not yet been resampled.
# The original data-file is available as a pickled file for fast reloading
# with Pandas, and as a CSV-file for broad compatibility.
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2018 by Magnus Erik Hvass Pedersen
#
########################################################################

import pandas as pd
import os
import download

########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_dir = "data/weather-denmark/"


# Full path for the pickled data-file. (Original data).
def path_original_data_pickle():
    return os.path.join(data_dir, "weather-denmark.pkl")


# Full path for the comma-separated text-file. (Original data).
def path_original_data_csv():
    return os.path.join(data_dir, "weather-denmark.csv")


# Full path for the resampled data as a pickled file.
def path_resampled_data_pickle():
    return os.path.join(data_dir, "weather-denmark-resampled.pkl")


# URL for the data-set on the internet.
data_url = "https://github.com/Hvass-Labs/weather-denmark/raw/master/weather-denmark.tar.gz"


# List of the cities in this data-set. These are cities in Denmark.
cities = ['Aalborg', 'Aarhus', 'Esbjerg', 'Odense', 'Roskilde']


########################################################################
# Private helper-functions.


def _date_string(x):
    """Convert two integers to a string for the date and time."""

    date = x[0]  # Date. Example: 19801231
    time = x[1]  # Time. Example: 1230

    return "{0}{1:04d}".format(date, time)


def _usaf_to_city(usaf):
    """
    The raw data-file uses USAF-codes to identify weather-stations.
    If you download another data-set from NCDC then you will have to
    change this function to use the USAF-codes in your new data-file.
    """

    table = \
        {
            60300: 'Aalborg',
            60700: 'Aarhus',
            60800: 'Esbjerg',
            61200: 'Odense',
            61700: 'Roskilde'
        }

    return table[usaf]


def _convert_raw_data(path):
    """
    This converts a raw data-file obtained from the NCDC database.
    This function may be useful as an inspiration if you want to
    download another raw data-file from NCDC, but you will have
    to modify this function to match the data you have downloaded.

    Note that you may also have to manually edit the raw data-file,
    e.g. because the header is not in a proper comma-separated format.
    """

    # The raw CSV-file uses various markers for "not-available" (NA).
    # (This is one of several oddities with NCDC's file-format.)
    na_values = ['999', '999.0', '999.9', '9999.9']

    # Use Pandas to load the comma-separated file.
    # Note that you may have to manually edit the file's header
    # to get this to load correctly.
    df_raw = pd.read_csv(path, sep=',', header=1,
                         index_col=False, na_values=na_values)

    # Create a new data-frame containing only the data
    # we are interested in.
    df = pd.DataFrame()

    # Get the city-name / weather-station name from the USAF code.
    df['City'] = df_raw['USAF  '].apply(_usaf_to_city)

    # Convert the integer date-time to a proper date-time object.
    datestr = df_raw[['Date    ', 'HrMn']].apply(_date_string, axis=1)
    df['DateTime'] = pd.to_datetime(datestr, format='%Y%m%d%H%M')

    # Get the data we are interested in.
    df['Temp'] = df_raw['Temp  ']
    df['Pressure'] = df_raw['Slp   ']
    df['WindSpeed'] = df_raw['Spd  ']
    df['WindDir'] = df_raw['Dir']

    # Set the city-name and date-time as the index.
    df.set_index(['City', 'DateTime'], inplace=True)

    # Save the new data-frame as a pickle for fast reloading.
    df.to_pickle(path_original_data_pickle())

    # Save the new data-frame as a CSV-file for general readability.
    df.to_csv(path_original_data_csv())

    return df


def _resample(df):
    """
    Resample the contents of a Pandas data-frame by first
    removing empty rows and columns, then up-sampling and
    interpolating the data for 1-minute intervals, and
    finally down-sampling to 60-minute intervals.
    """

    # Remove all empty rows and columns.
    df_res = df.dropna(axis=[0, 1], how='all')

    # Upsample so the time-series has data for every minute.
    df_res = df_res.resample('1T')

    # Fill in missing values.
    df_res = df_res.interpolate(method='time')

    # Downsample so the time-series has data for every hour.
    df_res = df_res.resample('60T')

    # Finalize the resampling. (Is this really necessary?)
    df_res = df_res.interpolate()

    # Remove all empty rows.
    df_res = df_res.dropna(how='all')

    return df_res


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.


def maybe_download_and_extract():
    """
    Download and extract the weather-data if the data-files don't
    already exist in the data_dir.
    """

    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)


def load_original_data():
    """
    Load and return the original data that has not been resampled.
    
    Note that this is not the raw data obtained from NCDC.
    It is a cleaned-up version of that data, as written by the
    function _convert_raw_data() above.
    """

    return pd.read_pickle(path_original_data_pickle())


def load_resampled_data():
    """
    Load and return the resampled weather-data.

    This has data-points at regular 60-minute intervals where
    missing data has been linearly interpolated.

    This uses a cache-file for saving and quickly reloading the data,
    so the original data is only resampled once.
    """

    # Path for the cache-file with the resampled data.
    path = path_resampled_data_pickle()

    # If the cache-file exists ...
    if os.path.exists(path):
        # Reload the cache-file.
        df = pd.read_pickle(path)
    else:
        # Otherwise resample the original data and save it in a cache-file.

        # Load the original data.
        df_org = load_original_data()

        # Split the original data into separate data-frames for each city.
        df_cities = [df_org.xs(city) for city in cities]

        # Resample the data for each city.
        df_resampled = [_resample(df_city) for df_city in df_cities]

        # Join the resampled data into a single data-frame.
        df = pd.concat(df_resampled, keys=cities,
                       axis=1, join='inner')

        # Save the resampled data in a cache-file for quick reloading.
        df.to_pickle(path)

    return df


########################################################################
