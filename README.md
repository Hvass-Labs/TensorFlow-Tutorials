# TensorFlow Tutorials

[Original repository on GitHub](https://github.com/Hvass-Labs/TensorFlow-Tutorials)

Original author is [Magnus Erik Hvass Pedersen](http://www.hvass-labs.org)

##  Donations

All this was made by a single person who did not receive any money for doing the work.
If you find it useful then [please donate securely using PayPal](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=PY9EUURN7GRUW).
Even a few dollars are appreciated. Thanks!

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

9. Video Data ([Notebook](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/09_Video_Data.ipynb))

10. Not available yet. Please [support this issue](https://github.com/tensorflow/tensorflow/issues/5036) on GitHub so we can get it done!

11. Adversarial Examples ([Notebook](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/11_Adversarial_Examples.ipynb))

12. Adversarial Noise for MNIST ([Notebook](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/12_Adversarial_Noise_MNIST.ipynb))

13. Visual Analysis ([Notebook](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/13_Visual_Analysis.ipynb))

14. DeepDream ([Notebook](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/14_DeepDream.ipynb))

15. Style Transfer ([Notebook](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/15_Style_Transfer.ipynb))

16. Reinforcement Learning ([Notebook](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/16_Reinforcement_Learning.ipynb))

## Videos

These tutorials are also available as [YouTube videos](https://www.youtube.com/playlist?list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ).

## Older Versions

Sometimes the source-code has changed from that shown in the YouTube videos. This may be due to
bug-fixes, improvements, or because code-sections are moved to separate files for easy re-use.

If you want to see the exact versions of the source-code that were used in the YouTube videos,
then you can [browse the history](https://github.com/Hvass-Labs/TensorFlow-Tutorials/commits/master)
of commits to the GitHub repository.

## Downloading

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

## Installation

There are different ways of installing and running TensorFlow. This section describes how I did it
for these tutorials. You may want to do it differently and you can search the internet for instructions.

If you are new to using Python and Linux, etc. then this may be challenging
to get working and you may need to do internet searches for error-messages, etc.
It will get easier with practice.

### Python Version 3.5 or Later

These tutorials were developed on Linux using **Python 3.5 / 3.6** (the [Anaconda](https://www.continuum.io/downloads) distribution) and [PyCharm](https://www.jetbrains.com/pycharm/).

There are reports that Python 2.7 gives error messages with these tutorials. Please make sure you are using **Python 3.5** or later!

### Environment

After installing [Anaconda](https://www.continuum.io/downloads), you should create a [conda environment](http://conda.pydata.org/docs/using/envs.html)
so you do not destroy your main installation in case you make a mistake somewhere:

    conda create --name tf python=3

Now you can switch to the new environment by running the following (on Linux):

    source activate tf

### Required Packages

The tutorials require several Python packages to be installed. The packages are listed in
[requirements.txt](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/requirements.txt)
First you need to edit this file and select whether you want to install the CPU or GPU
version of TensorFlow.

To install the required Python packages and dependencies you first have to activate the
conda-environment as described above, and then you run the following command
in a terminal:

    pip install -r requirements.txt

Note that the GPU-version of TensorFlow also requires the installation of various
NVIDIA drivers, which is not described here.

### Testing

You should now be able to run the tutorials in the Python Notebooks:

    cd ~/development/TensorFlow-Tutorials/  # Your installation directory.
    jupyter notebook

This should start a web-browser that shows the list of tutorials. Click on a tutorial to load it.

## License (MIT)

These tutorials and source-code are published under the [MIT License](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/LICENSE)
which allows very broad use for both academic and commercial purposes.

A few of the images used for demonstration purposes may be under copyright. These images are included under the "fair usage" laws.

You are very welcome to modify these tutorials and use them in your own projects.
Please keep a link to the [original repository](https://github.com/Hvass-Labs/TensorFlow-Tutorials).

