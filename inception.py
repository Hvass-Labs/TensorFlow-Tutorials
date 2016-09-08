########################################################################
#
# The Inception Model v3 for TensorFlow.
#
# This is a pre-trained Deep Neural Network for classifying images.
# You provide a filename for a jpeg-file which will be loaded and input
# to the Inception model, which will then output an array of numbers
# indicating how likely it is that the input-image is of each class.
#
# See the example code at the bottom of this file or in the
# accompanying Python Notebook.
#
# Implemented in Python 3.5 with TensorFlow v0.10.0rc0
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
data_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"

# Directory to store the downloaded data.
data_dir = "inception/"

# File containing the mappings between class-number and uid. (Downloaded)
path_uid_to_cls = os.path.join(data_dir, "imagenet_2012_challenge_label_map_proto.pbtxt")

# File containing the mappings between uid and string. (Downloaded)
path_uid_to_name = os.path.join(data_dir, "imagenet_synset_to_human_label_map.txt")

# File containing the TensorFlow graph definition. (Downloaded)
path_graph_def = os.path.join(data_dir, "classify_image_graph_def.pb")

########################################################################
# Names for tensors in the computational graph.

# Name of the tensor for feeding the input image - must be in jpeg-format.
tensor_name_jpeg = "DecodeJpeg/contents:0"

# Name of the tensor for the resized input image.
# This is used to retrieve the image after it has been resized.
tensor_name_resized_image = "ResizeBilinear:0"

# Name of the tensor for the output of the softmax-classifier.
tensor_name_softmax = "softmax:0"

########################################################################


def maybe_download():
    """
    Download the Inception model from the internet if it does not already
    exist in the data_dir. The file is about 85 MB.
    """

    print("Downloading Inception v3 Model ...")
    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)


########################################################################


class NameLookup:
    """
    Used for looking up the name associated with a class-number.
    This is used to print the name of a class instead of its number,
    e.g. "plant" or "horse".

    Maps between:
    - cls is the class-number as an integer between 1 and 1000 (inclusive).
    - uid is a class-id as a string from the ImageNet data-set, e.g. "n00017222".
    - name is the class-name as a string, e.g. "plant, flora, plant life"

    There are actually 1008 output classes of the Inception model
    but there are only 1000 named classes in these mapping-files.
    The remaining 8 output classes of the model should not be used.
    """

    def __init__(self):
        # Mappings between uid, cls and name are dicts, where insertions and
        # lookup have O(1) time-usage on average, but may be O(n) in worst case.
        self._uid_to_cls = {}   # Map from uid to cls.
        self._uid_to_name = {}  # Map from uid to name.
        self._cls_to_uid = {}   # Map from cls to uid.

        # Read the uid-to-name mappings from file.
        with open(file=path_uid_to_name, mode='r') as file:
            # Read all lines from the file.
            lines = file.readlines()

            for line in lines:
                # Remove newlines.
                line = line.replace("\n", "")

                # Split the line on tabs.
                elements = line.split("\t")

                # Get the uid.
                uid = elements[0]

                # Get the class-name.
                name = elements[1]

                # Insert into the lookup-dict.
                self._uid_to_name[uid] = name

        # Read the uid-to-cls mappings from file.
        with open(file=path_uid_to_cls, mode='r') as file:
            # Read all lines from the file.
            lines = file.readlines()

            for line in lines:
                # We assume the file is in the proper format,
                # so the following lines come in pairs. Other lines are ignored.

                if line.startswith("  target_class: "):
                    # This line must be the class-number as an integer.

                    # Split the line.
                    elements = line.split(": ")

                    # Get the class-number as an integer.
                    cls = int(elements[1])

                elif line.startswith("  target_class_string: "):
                    # This line must be the uid as a string.

                    # Split the line.
                    elements = line.split(": ")

                    # Get the uid as a string e.g. "n01494475"
                    uid = elements[1]

                    # Remove the enclosing "" from the string.
                    uid = uid[1:-2]

                    # Insert into the lookup-dicts for both ways between uid and cls.
                    self._uid_to_cls[uid] = cls
                    self._cls_to_uid[cls] = uid

    def uid_to_cls(self, uid):
        """
        Return the class-number as an integer for the given uid-string.
        """

        return self._uid_to_cls[uid]

    def uid_to_name(self, uid, only_first_name=False):
        """
        Return the class-name for the given uid string.

        Some class-names are lists of names, if you only want the first name,
        then set only_first_name=True.
        """

        # Lookup the name from the uid.
        name = self._uid_to_name[uid]

        # Only use the first name in the list?
        if only_first_name:
            name = name.split(",")[0]

        return name

    def cls_to_name(self, cls, only_first_name=False):
        """
        Return the class-name from the integer class-number.

        Some class-names are lists of names, if you only want the first name,
        then set only_first_name=True.
        """

        # Lookup the uid from the cls.
        uid = self._cls_to_uid[cls]

        # Lookup the name from the uid.
        name = self.uid_to_name(uid=uid, only_first_name=only_first_name)

        return name


########################################################################


class Inception:
    """
    The Inception model is a Deep Neural Network which has already been
    trained for classifying images into 1000 different categories.

    When you create a new instance of this class, the Inception model
    will be loaded and can be used immediately without training.
    """

    def __init__(self):
        # Mappings between class-numbers and class-names.
        # Used to print the class-name as a string e.g. "horse" or "plant".
        self.name_lookup = NameLookup()

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
            with tf.gfile.FastGFile(path_graph_def, 'rb') as file:
                # The graph-def is a saved copy of a TensorFlow graph.
                # First we need to create an empty graph-def.
                graph_def = tf.GraphDef()

                # Then we load the proto-buf file into the graph-def.
                graph_def.ParseFromString(file.read())

                # Finally we import the graph-def to the default TensorFlow graph.
                tf.import_graph_def(graph_def, name='')

                # Now self.graph holds the Inception model from the proto-buf file.

        # Get the output of the Inception model by looking up the tensor
        # with the appropriate name for the output of the softmax-classifier.
        self.y_pred = self.graph.get_tensor_by_name(tensor_name_softmax)

        # Get the tensor for the resized image that is input to the neural network.
        self.resized_image = self.graph.get_tensor_by_name(tensor_name_resized_image)

        # Create a TensorFlow session for executing the graph.
        self.session = tf.Session(graph=self.graph)

    def close(self):
        """
        Call this function when you are done using the Inception model.
        It closes the TensorFlow session to release its resources.
        """

        self.session.close()

    @staticmethod
    def _create_feed_dict(image_path):
        """
        Create and return a feed-dict with the image from the given file-path.
        The image must be a jpeg.
        """

        # Read the jpeg-image as an array of bytes.
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        # Create the feed-dict.
        feed_dict = {tensor_name_jpeg: image_data}

        return feed_dict

    def classify(self, image_path):
        """
        Use the Inception model to classify an image.

        The input is a file-path to a jpeg-image which will be loaded
        and fed to the neural network.

        The image will be resized automatically to 299 x 299 pixels,
        see the discussion in the accompanying Python Notebook.

        Returns an array of floats indicating how likely
        the Inception model thinks the image is of each given class.
        """

        # Create a feed-dict for the TensorFlow graph with the image.
        feed_dict = self._create_feed_dict(image_path=image_path)

        # Execute the TensorFlow session to get the predicted labels.
        pred = self.session.run(self.y_pred, feed_dict=feed_dict)

        # Reduce the array to a single dimension.
        pred = np.squeeze(pred)

        return pred

    def get_resized_image(self, image_path):
        """
        Input the given image to the Inception model and return
        the resized image. The resized image can be plotted so
        we can see what the neural network sees as its input.

        Returns a 3-dim array holding the image.
        """

        # Create a feed-dict for the TensorFlow graph with the image.
        feed_dict = self._create_feed_dict(image_path=image_path)

        # Execute the TensorFlow session to get the predicted labels.
        image = self.session.run(self.resized_image, feed_dict=feed_dict)

        # Remove the 1st dimension of the 4-dim tensor.
        image = image.squeeze(axis=0)

        # Normalize pixels to be between 0.0 and 1.0
        image = image.astype(float) / 255.0

        return image

    def print_scores(self, pred, k=10, only_first_name=True):
        """
        Print the scores (or probabilities) for the top-k predicted classes.

        Input is the predicted class-labels from the predict() function.

        Some class-names are lists of names, if you only want the first name,
        then set only_first_name=True.
        """

        # Get a sorted index for the pred-array.
        idx = pred.argsort()

        # The index is sorted lowest-to-highest values. Take the last k.
        top_k = idx[-k:]

        # Iterate the top-k classes in reversed order (i.e. highest first).
        for cls in reversed(top_k):
            # Lookup the class-name.
            name = self.name_lookup.cls_to_name(cls=cls, only_first_name=only_first_name)

            # Predicted score (or probability) for this class.
            score = pred[cls]

            # Print the score and class-name.
            print("{0:>6.2%} : {1}".format(score, name))


########################################################################
# Example usage.

if __name__ == '__main__':
    print(tf.__version__)

    # Download Inception model if not already done.
    maybe_download()

    # Load the Inception model so it is ready for classifying images.
    model = Inception()

    # Path for a jpeg-image that is included in the downloaded data.
    image_path = os.path.join(data_dir, 'cropped_panda.jpg')

    # Use the Inception model to classify the image.
    pred = model.classify(image_path=image_path)

    # Print the scores and names for the top-10 predictions.
    model.print_scores(pred=pred, k=10)

    # Close the TensorFlow session.
    model.close()

########################################################################
