########################################################################
#
# Functions for downloading the COCO data-set from the internet
# and loading it into memory. This data-set contains images and
# various associated data such as text-captions describing the images.
#
# http://cocodataset.org
#
# Implemented in Python 3.6
#
# Usage:
# 1) Call set_data_dir() to set the desired storage directory.
# 2) Call maybe_download_and_extract() to download the data-set
#    if it is not already located in the given data_dir.
# 3) Call load_records(train=True) and load_records(train=False)
#    to load the data-records for the training- and validation sets.
# 5) Use the returned data in your own program.
#
# Format:
# The COCO data-set contains a large number of images and various
# data for each image stored in a JSON-file.
# Functionality is provided for getting a list of image-filenames
# (but not actually loading the images) along with their associated
# data such as text-captions describing the contents of the images.
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

import json
import os
import download
from cache import cache

########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
# Use the function set_data_dir() to also update train_dir and val_dir.
data_dir = "data/coco/"

# Sub-directories for the training- and validation-sets.
train_dir = "data/coco/train2017"
val_dir = "data/coco/val2017"

# Base-URL for the data-sets on the internet.
data_url = "http://images.cocodataset.org/"


########################################################################
# Private helper-functions.

def _load_records(train=True):
    """
    Load the image-filenames and captions
    for either the training-set or the validation-set.
    """

    if train:
        # Training-set.
        filename = "captions_train2017.json"
    else:
        # Validation-set.
        filename = "captions_val2017.json"

    # Full path for the data-file.
    path = os.path.join(data_dir, "annotations", filename)

    # Load the file.
    with open(path, "r", encoding="utf-8") as file:
        data_raw = json.load(file)

    # Convenience variables.
    images = data_raw['images']
    annotations = data_raw['annotations']

    # Initialize the dict for holding our data.
    # The lookup-key is the image-id.
    records = dict()

    # Collect all the filenames for the images.
    for image in images:
        # Get the id and filename for this image.
        image_id = image['id']
        filename = image['file_name']

        # Initialize a new data-record.
        record = dict()

        # Set the image-filename in the data-record.
        record['filename'] = filename

        # Initialize an empty list of image-captions
        # which will be filled further below.
        record['captions'] = list()

        # Save the record using the the image-id as the lookup-key.
        records[image_id] = record

    # Collect all the captions for the images.
    for ann in annotations:
        # Get the id and caption for an image.
        image_id = ann['image_id']
        caption = ann['caption']

        # Lookup the data-record for this image-id.
        # This data-record should already exist from the loop above.
        record = records[image_id]

        # Append the current caption to the list of captions in the
        # data-record that was initialized in the loop above.
        record['captions'].append(caption)

    # Convert the records-dict to a list of tuples.
    records_list = [(key, record['filename'], record['captions'])
                    for key, record in sorted(records.items())]

    # Convert the list of tuples to separate tuples with the data.
    ids, filenames, captions = zip(*records_list)

    return ids, filenames, captions


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.


def set_data_dir(new_data_dir):
    """
    Set the base-directory for data-files and then
    set the sub-dirs for training and validation data.
    """

    # Ensure we update the global variables.
    global data_dir, train_dir, val_dir

    data_dir = new_data_dir
    train_dir = os.path.join(new_data_dir, "train2017")
    val_dir = os.path.join(new_data_dir, "val2017")


def maybe_download_and_extract():
    """
    Download and extract the COCO data-set if the data-files don't
    already exist in data_dir.
    """

    # Filenames to download from the internet.
    filenames = ["zips/train2017.zip", "zips/val2017.zip",
                 "annotations/annotations_trainval2017.zip"]

    # Download these files.
    for filename in filenames:
        # Create the full URL for the given file.
        url = data_url + filename

        print("Downloading " + url)

        download.maybe_download_and_extract(url=url, download_dir=data_dir)


def load_records(train=True):
    """
    Load the data-records for the data-set. This returns the image ids,
    filenames and text-captions for either the training-set or validation-set.
    
    This wraps _load_records() above with a cache, so if the cache-file already
    exists then it is loaded instead of processing the original data-file.
    
    :param train:
        Bool whether to load the training-set (True) or validation-set (False).

    :return: 
        ids, filenames, captions for the images in the data-set.
    """

    if train:
        # Cache-file for the training-set data.
        cache_filename = "records_train.pkl"
    else:
        # Cache-file for the validation-set data.
        cache_filename = "records_val.pkl"

    # Path for the cache-file.
    cache_path = os.path.join(data_dir, cache_filename)

    # If the data-records already exist in a cache-file then load it,
    # otherwise call the _load_records() function and save its
    # return-values to the cache-file so it can be loaded the next time.
    records = cache(cache_path=cache_path,
                    fn=_load_records,
                    train=train)

    return records

########################################################################
