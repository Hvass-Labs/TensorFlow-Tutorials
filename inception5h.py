########################################################################
#
# The Inception Model 5h for TensorFlow.
#
# This variant of the Inception model is easier to use for DeepDream
# and other imaging techniques. This is because it allows the input
# image to be any size, and the optimized images are also prettier.
#
# It is unclear which Inception model this implements because the
# Google developers have (as usual) neglected to document it.
# It is dubbed the 5h-model because that is the name of the zip-file,
# but it is apparently simpler than the v.3 model.
#
# See the Python Notebook for Tutorial #14 for an example usage.
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

# Internet URL for the tar-file with the Inception model.
# Note that this might change in the future and will need to be updated.
data_url = "http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"

# Directory to store the downloaded data.
data_dir = "inception/5h/"

# File containing the TensorFlow graph definition. (Downloaded)
path_graph_def = "tensorflow_inception_graph.pb"

########################################################################


def maybe_download():
    """
    Download the Inception model from the internet if it does not already
    exist in the data_dir. The file is about 50 MB.
    """

    print("Downloading Inception 5h Model ...")
    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)


########################################################################


class Inception5h:
    """
    The Inception model is a Deep Neural Network which has already been
    trained for classifying images into 1000 different categories.

    When you create a new instance of this class, the Inception model
    will be loaded and can be used immediately without training.
    """

    # Name of the tensor for feeding the input image.
    tensor_name_input_image = "input:0"

    # Names for some of the commonly used layers in the Inception model.
    layer_names = ['conv2d0', 'conv2d1', 'conv2d2',
                   'mixed3a', 'mixed3b',
                   'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e',
                   'mixed5a', 'mixed5b']

    def __init__(self):
        # Now load the Inception model from file. The way TensorFlow
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

                # Now self.graph holds the Inception model from the proto-buf file.

            # Get a reference to the tensor for inputting images to the graph.
            self.input = self.graph.get_tensor_by_name(self.tensor_name_input_image)

            # Get references to the tensors for the commonly used layers.
            self.layer_tensors = [self.graph.get_tensor_by_name(name + ":0") for name in self.layer_names]

    def create_feed_dict(self, image=None):
        """
        Create and return a feed-dict with an image.

        :param image:
            The input image is a 3-dim array which is already decoded.
            The pixels MUST be values between 0 and 255 (float or int).

        :return:
            Dict for feeding to the Inception graph in TensorFlow.
        """

        # Expand 3-dim array to 4-dim by prepending an 'empty' dimension.
        # This is because we are only feeding a single image, but the
        # Inception model was built to take multiple images as input.
        image = np.expand_dims(image, axis=0)

        # Image is passed in as a 3-dim array of raw pixel-values.
        feed_dict = {self.tensor_name_input_image: image}

        return feed_dict

    def get_gradient(self, tensor):
        """
        Get the gradient of the given tensor with respect to
        the input image. This allows us to modify the input
        image so as to maximize the given tensor.

        For use in e.g. DeepDream and Visual Analysis.

        :param tensor:
            The tensor whose value we want to maximize
            by changing the input image.

        :return:
            Gradient for the tensor with regard to the input image.
        """

        # Set the graph as default so we can add operations to it.
        with self.graph.as_default():
            # Square the tensor-values.
            # You can try and remove this to see the effect.
            tensor = tf.square(tensor)

            # Average the tensor so we get a single scalar value.
            tensor_mean = tf.reduce_mean(tensor)

            # Use TensorFlow to automatically create a mathematical
            # formula for the gradient using the chain-rule of
            # differentiation.
            gradient = tf.gradients(tensor_mean, self.input)[0]

        return gradient

########################################################################
