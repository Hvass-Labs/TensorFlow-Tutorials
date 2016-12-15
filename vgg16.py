########################################################################
#
# The pre-trained VGG16 Model for TensorFlow.
#
# This model seems to produce better-looking images in Style Transfer
# than the Inception 5h model that otherwise works well for DeepDream.
#
# See the Python Notebook for Tutorial #15 for an example usage.
#
# Implemented in Python 3.5 with TensorFlow v0.11.0rc0
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import numpy as np
import tensorflow as tf
import download
import os

########################################################################
# Various directories and file-names.

# The pre-trained VGG16 model is taken from this tutorial:
# https://github.com/pkmital/CADL/blob/master/session-4/libs/vgg16.py

# The class-names are available in the following URL:
# https://s3.amazonaws.com/cadl/models/synset.txt

# Internet URL for the file with the VGG16 model.
# Note that this might change in the future and will need to be updated.
data_url = "https://s3.amazonaws.com/cadl/models/vgg16.tfmodel"

# Directory to store the downloaded data.
data_dir = "vgg16/"

# File containing the TensorFlow graph definition. (Downloaded)
path_graph_def = "vgg16.tfmodel"

########################################################################


def maybe_download():
    """
    Download the VGG16 model from the internet if it does not already
    exist in the data_dir. WARNING! The file is about 550 MB.
    """

    print("Downloading VGG16 Model ...")

    # The file on the internet is not stored in a compressed format.
    # This function should not extract the file when it does not have
    # a relevant filename-extensions such as .zip or .tar.gz
    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)


########################################################################


class VGG16:
    """
    The VGG16 model is a Deep Neural Network which has already been
    trained for classifying images into 1000 different categories.

    When you create a new instance of this class, the VGG16 model
    will be loaded and can be used immediately without training.
    """

    # Name of the tensor for feeding the input image.
    tensor_name_input_image = "images:0"

    # Names of the tensors for the dropout random-values..
    tensor_name_dropout = 'dropout/random_uniform:0'
    tensor_name_dropout1 = 'dropout_1/random_uniform:0'

    # Names for the convolutional layers in the model for use in Style Transfer.
    layer_names = ['conv1_1/conv1_1', 'conv1_2/conv1_2',
                   'conv2_1/conv2_1', 'conv2_2/conv2_2',
                   'conv3_1/conv3_1', 'conv3_2/conv3_2', 'conv3_3/conv3_3',
                   'conv4_1/conv4_1', 'conv4_2/conv4_2', 'conv4_3/conv4_3',
                   'conv5_1/conv5_1', 'conv5_2/conv5_2', 'conv5_3/conv5_3']

    def __init__(self):
        # Now load the model from file. The way TensorFlow
        # does this is confusing and requires several steps.

        # Create a new TensorFlow computational graph.
        self.graph = tf.Graph()

        # Set the new graph as the default.
        with self.graph.as_default():

            # TensorFlow graphs are saved to disk as so-called Protocol Buffers
            # aka. proto-bufs which is a file-format that works on multiple
            # platforms. In this case it is saved as a binary file.

            # Open the graph-def file for binary reading.
            path = os.path.join(data_dir, path_graph_def)
            with tf.gfile.FastGFile(path, 'rb') as file:
                # The graph-def is a saved copy of a TensorFlow graph.
                # First we need to create an empty graph-def.
                graph_def = tf.GraphDef()

                # Then we load the proto-buf file into the graph-def.
                graph_def.ParseFromString(file.read())

                # Finally we import the graph-def to the default TensorFlow graph.
                tf.import_graph_def(graph_def, name='')

                # Now self.graph holds the VGG16 model from the proto-buf file.

            # Get a reference to the tensor for inputting images to the graph.
            self.input = self.graph.get_tensor_by_name(self.tensor_name_input_image)

            # Get references to the tensors for the commonly used layers.
            self.layer_tensors = [self.graph.get_tensor_by_name(name + ":0") for name in self.layer_names]

    def get_layer_tensors(self, layer_ids):
        """
        Return a list of references to the tensors for the layers with the given id's.
        """

        return [self.layer_tensors[idx] for idx in layer_ids]

    def get_layer_names(self, layer_ids):
        """
        Return a list of names for the layers with the given id's.
        """

        return [self.layer_names[idx] for idx in layer_ids]

    def get_all_layer_names(self, startswith=None):
        """
        Return a list of all the layers (operations) in the graph.
        The list can be filtered for names that start with the given string.
        """

        # Get a list of the names for all layers (operations) in the graph.
        names = [op.name for op in self.graph.get_operations()]

        # Filter the list of names so we only get those starting with
        # the given string.
        if startswith is not None:
            names = [name for name in names if name.startswith(startswith)]

        return names

    def create_feed_dict(self, image):
        """
        Create and return a feed-dict with an image.

        :param image:
            The input image is a 3-dim array which is already decoded.
            The pixels MUST be values between 0 and 255 (float or int).

        :return:
            Dict for feeding to the graph in TensorFlow.
        """

        # Expand 3-dim array to 4-dim by prepending an 'empty' dimension.
        # This is because we are only feeding a single image, but the
        # VGG16 model was built to take multiple images as input.
        image = np.expand_dims(image, axis=0)

        if False:
            # In the original code using this VGG16 model, the random values
            # for the dropout are fixed to 1.0.
            # Experiments suggest that it does not seem to matter for
            # Style Transfer, and this causes an error with a GPU.
            dropout_fix = 1.0

            # Create feed-dict for inputting data to TensorFlow.
            feed_dict = {self.tensor_name_input_image: image,
                         self.tensor_name_dropout: [[dropout_fix]],
                         self.tensor_name_dropout1: [[dropout_fix]]}
        else:
            # Create feed-dict for inputting data to TensorFlow.
            feed_dict = {self.tensor_name_input_image: image}

        return feed_dict

########################################################################
