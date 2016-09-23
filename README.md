# TensorFlow Tutorials

[Original repository on GitHub](https://github.com/Hvass-Labs/TensorFlow-Tutorials)

Original author is [Magnus Erik Hvass Pedersen](http://www.hvass-labs.org)

## Introduction

* These tutorials are intended for beginners in Deep Learning and TensorFlow.
* Each tutorial covers a single topic.
* The source-code is well-documented.
* There is a [YouTube video](https://www.youtube.com/playlist?list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ) for each tutorial.

## Tutorials

1. Simple Linear Model ([Notebook](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/01_Simple_Linear_Model.ipynb))

2. Convolutional Neural Network ([Notebook](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb))

3. Pretty Tensor ([Notebook](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/03_PrettyTensor.ipynb))

4. Save & Restore ([Notebook](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/04_Save_Restore.ipynb))

5. Ensemble Learning ([Notebook](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/05_Ensemble_Learning.ipynb))

6. CIFAR-10 ([Notebook](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/06_CIFAR-10.ipynb))

7. Inception Model ([Notebook](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/07_Inception_Model.ipynb))

8. Transfer Learning ([Notebook](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/08_Transfer_Learning.ipynb))

## Videos

These tutorials are also available as [YouTube videos](https://www.youtube.com/playlist?list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ).

## Older Versions

Sometimes the source-code has changed from that shown in the YouTube videos. This may be due to
bug-fixes, improvements, or because code-sections are moved to separate files for easy re-use.

If you want to see the exact versions of the source-code that were used in the YouTube videos,
then you can [browse the history](https://github.com/Hvass-Labs/TensorFlow-Tutorials/commits/master)
of commits to the GitHub repository.

## Installation

Some of the Python Notebooks use source-code located in different files to allow for easy re-use
across multiple tutorials. It is therefore recommended that you download the whole repository
from GitHub, instead of just downloading the individual Python Notebooks.

### Git

The easiest way to download and install these tutorials is by using git from the command-line:

    git clone https://github.com/Hvass-Labs/TensorFlow-Tutorials.git

This will create the directory `TensorFlow-Tutorials` and download all the files to it.

This also makes it easy to update the tutorials, simply by executing this command inside that directory:

    git pull

### Zip-File

You can also [download](https://github.com/Hvass-Labs/TensorFlow-Tutorials/archive/master.zip)
the contents of the GitHub repository as a Zip-file and extract it manually.

## Requirements

There are different ways of installing and running TensorFlow. This section describes how I did it
for these tutorials. You may want to do it differently and you can search the internet for instructions.

These tutorials were developed on Linux using [Anaconda](https://www.continuum.io/downloads) with Python 3.5 and [PyCharm](https://www.jetbrains.com/pycharm/).

After installing Anaconda, you should create a [conda environment](http://conda.pydata.org/docs/using/envs.html)
so you do not destroy your main installation in case you make a mistake somewhere:

    conda create --name tf python=3

Now you can switch to the new environment by running the following (on Linux):

    source activate tf

Some of these tutorials use [scikit-learn](http://scikit-learn.org/stable/install.html)
which can be installed in your new conda environment as follows. This also installs
NumPy and other dependencies:

    conda install scikit-learn

You may also need to install Jupyter Notebook and matplotlib:

    conda install jupyter matplotlib

Now you have to install TensorFlow. This procedure might change in the future. At the time of this writing,
the most recent TensorFlow version was 0.10.0. It comes in different builds depending on your needs.
I need the Python 3.5 build for a Linux PC with only a CPU (no GPU). So I look at the [list of builds](https://www.tensorflow.org/versions/master/get_started/os_setup.html)
and find the appropriate link which in my case is:
 
    pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp35-cp35m-linux_x86_64.whl

It is much more complicated to install the GPU-version because you also need various NVIDIA drivers.
That is not described here.

You should now be able to run the tutorials in the Python Notebooks:

    cd ~/development/TensorFlow-Tutorials/  # Your installation directory.
    jupyter notebook

This should start a web-browser that shows the list of tutorials. Click on a tutorial to load it.

If you are new to using Python and Linux, etc. then this may be challenging
to get working and you may need to do internet searches for error-messages, etc.
It will get easier with practice.

## License (MIT)

These tutorials and source-code are published under the [MIT License](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/LICENSE)
which allows very broad use for both academic and commercial purposes.

A few of the images used for demonstration purposes may be under copyright. These images are included under the "fair usage" laws.

You are very welcome to modify these tutorials and use them in your own projects.
Please keep a link to the [original repository](https://github.com/Hvass-Labs/TensorFlow-Tutorials).

