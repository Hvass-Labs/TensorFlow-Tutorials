########################################################################
#
# Reinforcement Learning (Q-Learning) for Atari Games
#
# How to run:
#
# To train a Neural Network for playing the Atari game Breakout,
# run the following command in a terminal window.
#
# python reinforcement-learning.py --env 'Breakout-v0' --training
#
# The agent should start to improve after a few hours, but a full
# training run required 150 hours on a 2.6 GHz CPU and GTX 1070 GPU.
#
# The hyper-parameters were tuned for Breakout and did not work
# quite as well for SpaceInvaders. Can you find better parameters?
#
# Once the Neural Network has been trained, you can test it and
# watch it play the game by running this command in the terminal:
#
# python reinforcement-learning.py --env 'Breakout-v0' --render --episodes 2
#
# Requirements:
#
# - Python 3.6 (Python 2.7 may not work)
# - TensorFlow 1.1.0
# - OpenAI Gym 0.8.1
# - PrettyTensor 0.7.4 (not required if you use tf.layers instead)
#
# Summary:
#
# This program implements a variant of Reinforcement Learning known as
# Q-learning. Imagine that we have an agent that must take actions in
# some environment so as to maximize the cumulative reward over its life.
# The agent sees the state of the game-environment through images
# which are sent through a Neural Network in TensorFlow, so as to
# estimate which action is most likely to maximize the cumulative
# reward of all future actions. These action-values are also called
# Q-values. If the Q-values are known in advance, then the agent merely
# has to select the action corresponding to the highest Q-value in
# each state of the game. But the Q-values are not known in advance
# and must be learnt while the agent is playing the game.
# This is done by initializing all Q-values to zero and then having
# the agent take random actions. Whenever the agent obtains a reward,
# the estimated Q-values can be updated with the new information.
# The agent gradually learns to play the game better and better
# because the Neural Network becomes better at estimating the Q-values.
# But this process is very slow and the basic algorithm implemented
# here typically requires 100 million steps in the game-environment,
# although it will typically start to show improvement much sooner.
#
# Main classes:
#
# - MotionTracer:
#
#   This takes raw images from the game-environment and processes them.
#   The output is called a state and consists of two images of equal size:
#   (1) The last image from the game-environment, resized and gray-scaled.
#   (2) A motion-trace that shows the recent trajectories of objects.
#
# - ReplayMemory:
#
#   Successive image-frames of the game-environment are almost identical.
#   If we train the Neural Network to estimate Q-values from a small
#   number of successive image-frames, then it cannot learn to distinguish
#   important features and the training becomes unstable. For the basic
#   Q-learning algorithm we need many thousand states from the game-environment
#   in order to learn important features so the Q-values can be estimated.
#
# - NeuralNetwork:
#
#   This implements a Neural Network for estimating Q-values. It takes as
#   input a state of the game-environment that was output by the Motion Tracer,
#   and then the Neural Network outputs the estimated Q-values that indicate
#   the cumulative reward of taking each action for a given state of the game.
#
# - Agent:
#
#   This implements the agent that plays games. It loads an Atari-game from
#   OpenAI Gym and inputs the game-images to the Motion Tracer, which in turn
#   outputs a state that is input to the Neural Network, which estimates the
#   Q-values that are used for selecting the next action. The agent then
#   takes a step in the game-environment. During training, the data is added
#   to the Replay Memory and when it is sufficiently full, an optimization run
#   is performed so as to improve the Neural Network's ability to estimate
#   Q-values. This procedure is repeated many, many times until the Neural
#   Network is sufficiently accurate at estimating Q-values.
#
# The Q-Value Formula:
#
# The Q-values for a given state is a vector with a value for each possible
# action, indicating the total future reward that can be had by taking each
# action. The Q-values are initialized to roughly zero and must then be
# improved iteratively when new information becomes available.
#
# We know which action was taken in the current step and what the observed
# reward was, so the estimated Q-value can be improved with this information.
# The Q-value estimates the total cumulative reward for all future steps, which
# is why we use the max Q-value for the next step.
#
# The formula for updating Q-values is implemented in the ReplayMemory-class
# in the function update_all_q_values(), which does a complete backwards-sweep
# through the Replay Memory. The formula for updating the Q-values is:
#
# Q-value for this state and action = observed reward for the current step
#                            + discount factor * max Q-value for next step
#
# The discount factor is a number slightly below 1.0 (e.g. 0.97) which causes
# distant future rewards to have a smaller effect on the Q-values. This means
# that if the reward is the same, then it is considered more valuable to get
# the reward sooner rather than later.
#
# Pseudo-Code:
#
# There are many lines of source-code required to implement all this, but the
# main ideas of the algorithm can be described more simply in pseudo-code:
#
# 1) Initialize all Q-values to roughly zero.
#    We use a Neural Network to estimate the Q-values, so this means
#    we have to initialize the Neural Network with small random weights.
#
# 2) Reset the game-environment and Motion Tracer.
#
# 3) Get the state from the Motion Tracer which consists of two gray-scale
#    images. The first is the image of the game-environment and the second
#    is a motion-trace showing recent movements in the game-environment.
#
# 4) Input the state to the Neural Network to estimate the Q-values.
#
# 5) Either take a random action with probability epsilon, or take the
#    action with the highest Q-value. This is called the epsilon-greedy policy.
#
# 6) Add the state, action and observed reward to the Replay Memory.
#
# 7) When the Replay Memory is sufficiently full, first perform a full
#    backwards-sweep to update all the Q-values with the observed rewards.
#
#    Then perform an optimization run of the Neural Network.
#    This takes random batches of data from the Replay Memory and uses them
#    for training the Neural Network to become better at estimating Q-values.
#
#    Save a checkpoint for the Neural Network so we can reload it later.
#
# 8) Input the recent image of the game-environment to the Motion Tracer
#    and repeat from step (3).
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2017 by Magnus Erik Hvass Pedersen
#
########################################################################

import numpy as np
import tensorflow as tf
import gym
import scipy.ndimage
import sys
import os
import time
import csv
import argparse
import download

########################################################################
# File-paths are global variables for convenience so they don't
# have to be passed around between all the objects.
# You should first set checkpoint_base_dir to whichever you like,
# then call the function update_paths(env_name) to update all the paths.
# This should be done before you create the Agent and NeuralNetwork etc.

# Default base-directory for the checkpoints and log-files.
# The environment-name will be appended to this.
checkpoint_base_dir = 'checkpoints_tutorial16/'

# Combination of base-dir and environment-name.
checkpoint_dir = None

# Full path for the log-file for rewards.
log_reward_path = None

# Full path for the log-file for Q-values.
log_q_values_path = None


def update_paths(env_name):
    """
    Update the path-names for the checkpoint-dir and log-files.
    
    Call this after you have changed checkpoint_base_dir and
    before you create the Neural Network.
    
    :param env_name:
        Name of the game-environment you will use in OpenAI Gym.
    """

    global checkpoint_dir
    global log_reward_path
    global log_q_values_path

    # Add the environment-name to the checkpoint-dir.
    checkpoint_dir = os.path.join(checkpoint_base_dir, env_name)

    # Create the checkpoint-dir if it does not already exist.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # File-path for the log-file for episode rewards.
    log_reward_path = os.path.join(checkpoint_dir, "log_reward.txt")

    # File-path for the log-file for Q-values.
    log_q_values_path = os.path.join(checkpoint_dir, "log_q_values.txt")

########################################################################
# Download TensorFlow checkpoints.

# URL's for the checkpoint-files.
_checkpoint_url = {
    "Breakout-v0": "http://hvass-labs.org/projects/tensorflow/tutorial16/Breakout-v0.tar.gz",
    "SpaceInvaders-v0": "http://hvass-labs.org/projects/tensorflow/tutorial16/SpaceInvaders-v0.tar.gz"
}


def maybe_download_checkpoint(env_name):
    """
    Download and extract the TensorFlow checkpoint for the given
    environment-name, if it does not already exist in checkpoint_base_dir.
    You should first set this dir and call update_paths().
    """

    # Get the url for the game-environment.
    url = _checkpoint_url[env_name]

    # Download and extract the file if it does not already exist.
    download.maybe_download_and_extract(url=url,
                                        download_dir=checkpoint_base_dir)

########################################################################
# Classes used for logging data during training.


class Log:
    """
    Base-class for logging data to a text-file during training.

    It is possible to use TensorFlow / TensorBoard for this,
    but it is quite awkward to implement, as it was intended
    for logging variables and other aspects of the TensorFlow graph.
    We want to log the reward and Q-values which are not in that graph.
    """

    def __init__(self, file_path):
        """Set the path for the log-file. Nothing is saved or loaded yet."""

        # Path for the log-file.
        self.file_path = file_path

        # Data to be read from the log-file by the _read() function.
        self.count_episodes = None
        self.count_states = None
        self.data = None

    def _write(self, count_episodes, count_states, msg):
        """
        Write a line to the log-file. This is only called by sub-classes.
        
        :param count_episodes:
            Counter for the number of episodes processed during training.

        :param count_states: 
            Counter for the number of states processed during training.

        :param msg:
            Message to write in the log.
        """

        with open(file=self.file_path, mode='a', buffering=1) as file:
            msg_annotated = "{0}\t{1}\t{2}\n".format(count_episodes, count_states, msg)
            file.write(msg_annotated)

    def _read(self):
        """
        Read the log-file into memory so it can be plotted.

        It sets self.count_episodes, self.count_states and self.data
        """

        # Open and read the log-file.
        with open(self.file_path) as f:
            reader = csv.reader(f, delimiter="\t")
            self.count_episodes, self.count_states, *data = zip(*reader)

        # Convert the remaining log-data to a NumPy float-array.
        self.data = np.array(data, dtype='float')


class LogReward(Log):
    """Log the rewards obtained for episodes during training."""

    def __init__(self):
        # These will be set in read() below.
        self.episode = None
        self.mean = None

        # Super-class init.
        Log.__init__(self, file_path=log_reward_path)

    def write(self, count_episodes, count_states, reward_episode, reward_mean):
        """
        Write the episode and mean reward to file.
        
        :param count_episodes:
            Counter for the number of episodes processed during training.

        :param count_states: 
            Counter for the number of states processed during training.

        :param reward_episode:
            Reward for one episode.

        :param reward_mean:
            Mean reward for the last e.g. 30 episodes.
        """

        msg = "{0:.1f}\t{1:.1f}".format(reward_episode, reward_mean)
        self._write(count_episodes=count_episodes, count_states=count_states, msg=msg)

    def read(self):
        """
        Read the log-file into memory so it can be plotted.

        It sets self.count_episodes, self.count_states, self.episode and self.mean
        """

        # Read the log-file using the super-class.
        self._read()

        # Get the episode reward.
        self.episode = self.data[0]

        # Get the mean reward.
        self.mean = self.data[1]


class LogQValues(Log):
    """Log the Q-Values during training."""

    def __init__(self):
        # These will be set in read() below.
        self.min = None
        self.mean = None
        self.max = None
        self.std = None

        # Super-class init.
        Log.__init__(self, file_path=log_q_values_path)

    def write(self, count_episodes, count_states, q_values):
        """
        Write basic statistics for the Q-values to file.

        :param count_episodes:
            Counter for the number of episodes processed during training.

        :param count_states: 
            Counter for the number of states processed during training.

        :param q_values:
            Numpy array with Q-values from the replay-memory.
        """

        msg = "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(np.min(q_values),
                                                          np.mean(q_values),
                                                          np.max(q_values),
                                                          np.std(q_values))

        self._write(count_episodes=count_episodes,
                    count_states=count_states,
                    msg=msg)

    def read(self):
        """
        Read the log-file into memory so it can be plotted.

        It sets self.count_episodes, self.count_states, self.min / mean / max / std.
        """

        # Read the log-file using the super-class.
        self._read()

        # Get the logged statistics for the Q-values.
        self.min = self.data[0]
        self.mean = self.data[1]
        self.max = self.data[2]
        self.std = self.data[3]

########################################################################


def print_progress(msg):
    """
    Print progress on a single line and overwrite the line.
    Used during optimization.
    """

    sys.stdout.write("\r" + msg)
    sys.stdout.flush()

########################################################################
# A state is basically just a multi-dimensional array that is being
# input to the Neural Network. The state consists of pre-processed images
# from the game-environment. We will just convert the game-images to
# gray-scale and resize them to roughly half their size. This is mainly
# so we can save memory-space in the Replay Memory further below.
# The original DeepMind paper used game-states consisting of 4 frames of
# game-images that were gray-scaled, resized to 110 x 84 pixels, and then
# cropped to 84 x 84 pixels because their implementation only supported this.

# Height of each image-frame in the state.
state_height = 105

# Width of each image-frame in the state.
state_width = 80

# Size of each image in the state.
state_img_size = np.array([state_height, state_width])

# Number of images in the state.
state_channels = 2

# Shape of the state-array.
state_shape = [state_height, state_width, state_channels]

########################################################################
# Functions and classes for processing images from the game-environment
# and converting them into a state.


def _rgb_to_grayscale(image):
    """
    Convert an RGB-image into gray-scale using a formula from Wikipedia:
    https://en.wikipedia.org/wiki/Grayscale
    """

    # Get the separate colour-channels.
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Convert to gray-scale using the Wikipedia formula.
    img_gray = 0.2990 * r + 0.5870 * g + 0.1140 * b

    return img_gray


def _pre_process_image(image):
    """Pre-process a raw image from the game-environment."""

    # Convert image to gray-scale.
    img = _rgb_to_grayscale(image)

    # Resize to the desired size using SciPy for convenience.
    img = scipy.misc.imresize(img, size=state_img_size, interp='bicubic')

    return img


class MotionTracer:
    """
    Used for processing raw image-frames from the game-environment.

    The image-frames are converted to gray-scale, resized, and then
    the background is removed using filtering of the image-frames
    so as to detect motions.

    This is needed because a single image-frame of the game environment
    is insufficient to determine the direction of moving objects.
    
    The original DeepMind implementation used the last 4 image-frames
    of the game-environment to allow the Neural Network to learn how
    to detect motion. This implementation could make it a little easier
    for the Neural Network to learn how to detect motion, but it has
    only been tested on Breakout and Space Invaders, and may not work
    for games with more complicated graphics such as Doom. This remains
    to be tested.
    """

    def __init__(self, image, decay=0.75):
        """
        
        :param image:
            First image from the game-environment,
            used for resetting the motion detector.

        :param decay:
            Parameter for how long the tail should be on the motion-trace.
            This is a float between 0.0 and 1.0 where higher values means
            the trace / tail is longer.
        """

        # Pre-process the image and save it for later use.
        # The input image may be 8-bit integers but internally
        # we need to use floating-point to avoid image-noise
        # caused by recurrent rounding-errors.
        img = _pre_process_image(image=image)
        self.last_input = img.astype(np.float)

        # Set the last output to zero.
        self.last_output = np.zeros_like(img)

        self.decay = decay

    def process(self, image):
        """Process a raw image-frame from the game-environment."""

        # Pre-process the image so it is gray-scale and resized.
        img = _pre_process_image(image=image)

        # Subtract the previous input. This only leaves the
        # pixels that have changed in the two image-frames.
        img_dif = img - self.last_input

        # Copy the contents of the input-image to the last input.
        self.last_input[:] = img[:]

        # If the pixel-difference is greater than a threshold then
        # set the output pixel-value to the highest value (white),
        # otherwise set the output pixel-value to the lowest value (black).
        # So that we merely detect motion, and don't care about details.
        img_motion = np.where(np.abs(img_dif) > 20, 255.0, 0.0)

        # Add some of the previous output. This recurrent formula
        # is what gives the trace / tail.
        output = img_motion + self.decay * self.last_output

        # Ensure the pixel-values are within the allowed bounds.
        output = np.clip(output, 0.0, 255.0)

        # Set the last output.
        self.last_output = output

        return output

    def get_state(self):
        """
        Get a state that can be used as input to the Neural Network.

        It is basically just the last input and the last output of the
        motion-tracer. This means it is the last image-frame of the
        game-environment, as well as the motion-trace. This shows
        the current location of all the objects in the game-environment
        as well as trajectories / traces of where they have been.
        """

        # Stack the last input and output images.
        state = np.dstack([self.last_input, self.last_output])

        # Convert to 8-bit integer.
        # This is done to save space in the replay-memory.
        state = state.astype(np.uint8)

        return state

########################################################################


class ReplayMemory:
    """
    The replay-memory holds many previous states of the game-environment.
    This helps stabilize training of the Neural Network because the data
    is more diverse when sampled over thousands of different states.
    """

    def __init__(self, size, num_actions, discount_factor=0.97):
        """
        
        :param size:
            Capacity of the replay-memory. This is the number of states.

        :param num_actions:
            Number of possible actions in the game-environment. 

        :param discount_factor:
            Discount-factor used for updating Q-values.
        """

        # Array for the previous states of the game-environment.
        self.states = np.zeros(shape=[size] + state_shape, dtype=np.uint8)

        # Array for the Q-values corresponding to the states.
        self.q_values = np.zeros(shape=[size, num_actions], dtype=np.float)

        # Array for the Q-values before being updated.
        # This is used to compare the Q-values before and after the update.
        self.q_values_old = np.zeros(shape=[size, num_actions], dtype=np.float)

        # Actions taken for each of the states in the memory.
        self.actions = np.zeros(shape=size, dtype=np.int)

        # Rewards observed for each of the states in the memory.
        self.rewards = np.zeros(shape=size, dtype=np.float)

        # Whether the life had ended in each state of the game-environment.
        self.end_life = np.zeros(shape=size, dtype=np.bool)

        # Whether the episode had ended (aka. game over) in each state.
        self.end_episode = np.zeros(shape=size, dtype=np.bool)

        # Estimation errors for the Q-values. This is used to balance
        # the sampling of batches for training the Neural Network,
        # so we get a balanced combination of states with high and low
        # estimation errors for their Q-values.
        self.estimation_errors = np.zeros(shape=size, dtype=np.float)

        # Capacity of the replay-memory as the number of states.
        self.size = size

        # Discount-factor for calculating Q-values.
        self.discount_factor = discount_factor

        # Reset the number of used states in the replay-memory.
        self.num_used = 0

        # Threshold for splitting between low and high estimation errors.
        self.error_threshold = 0.1

    def is_full(self):
        """Return boolean whether the replay-memory is full."""
        return self.num_used == self.size

    def used_fraction(self):
        """Return the fraction of the replay-memory that is used."""
        return self.num_used / self.size

    def reset(self):
        """Reset the replay-memory so it is empty."""
        self.num_used = 0

    def add(self, state, q_values, action, reward, end_life, end_episode):
        """
        Add an observed state from the game-environment, along with the
        estimated Q-values, action taken, observed reward, etc. 
        
        :param state:
            Current state of the game-environment.
            This is the output of the MotionTracer-class.

        :param q_values: 
            The estimated Q-values for the state.

        :param action: 
            The action taken by the agent in this state of the game.

        :param reward:
            The reward that was observed from taking this action
            and moving to the next state.

        :param end_life:
            Boolean whether the agent has lost a life in this state.
         
        :param end_episode: 
            Boolean whether the agent has lost all lives aka. game over
            aka. end of episode.
        """

        if not self.is_full():
            # Index into the arrays for convenience.
            k = self.num_used

            # Increase the number of used elements in the replay-memory.
            self.num_used += 1

            # Store all the values in the replay-memory.
            self.states[k] = state
            self.q_values[k] = q_values
            self.actions[k] = action
            self.end_life[k] = end_life
            self.end_episode[k] = end_episode

            # Note that the reward is limited. This is done to stabilize
            # the training of the Neural Network.
            self.rewards[k] = np.clip(reward, -1.0, 1.0)

    def update_all_q_values(self):
        """
        Update all Q-values in the replay-memory.
        
        When states and Q-values are added to the replay-memory, the
        Q-values have been estimated by the Neural Network. But we now
        have more data available that we can use to improve the estimated
        Q-values, because we now know which actions were taken and the
        observed rewards. We sweep backwards through the entire replay-memory
        to use the observed data to improve the estimated Q-values.
        """

        # Copy old Q-values so we can print their statistics later.
        # Note that the contents of the arrays are copied.
        self.q_values_old[:] = self.q_values[:]

        # Process the replay-memory backwards and update the Q-values.
        # This loop could be implemented entirely in NumPy for higher speed,
        # but it is probably only a small fraction of the overall time usage,
        # and it is much easier to understand when implemented like this.
        for k in reversed(range(self.num_used-1)):
            # Get the data for the k'th state in the replay-memory.
            action = self.actions[k]
            reward = self.rewards[k]
            end_life = self.end_life[k]
            end_episode = self.end_episode[k]

            # Calculate the Q-value for the action that was taken in this state.
            if end_life or end_episode:
                # If the agent lost a life or it was game over / end of episode,
                # then the value of taking the given action is just the reward
                # that was observed in this single step. This is because the
                # Q-value is defined as the discounted value of all future game
                # steps in a single life of the agent. When the life has ended,
                # there will be no future steps.
                action_value = reward
            else:
                # Otherwise the value of taking the action is the reward that
                # we have observed plus the discounted value of future rewards
                # from continuing the game. We use the estimated Q-values for
                # the following state and take the maximum, because we will
                # generally take the action that has the highest Q-value.
                action_value = reward + self.discount_factor * np.max(self.q_values[k + 1])

            # Error of the Q-value that was estimated using the Neural Network.
            self.estimation_errors[k] = abs(action_value - self.q_values[k, action])

            # Update the Q-value with the better estimate.
            self.q_values[k, action] = action_value

        self.print_statistics()

    def prepare_sampling_prob(self, batch_size=128):
        """
        Prepare the probability distribution for random sampling of states
        and Q-values for use in training of the Neural Network.

        The probability distribution is just a simple binary split of the
        replay-memory based on the estimation errors of the Q-values.
        The idea is to create a batch of samples that are balanced somewhat
        evenly between Q-values that the Neural Network already knows how to
        estimate quite well because they have low estimation errors, and
        Q-values that are poorly estimated by the Neural Network because
        they have high estimation errors.
        
        The reason for this balancing of Q-values with high and low estimation
        errors, is that if we train the Neural Network mostly on data with
        high estimation errors, then it will tend to forget what it already
        knows and hence become over-fit so the training becomes unstable.
        """

        # Get the errors between the Q-values that were estimated using
        # the Neural Network, and the Q-values that were updated with the
        # reward that was actually observed when an action was taken.
        err = self.estimation_errors[0:self.num_used]

        # Create an index of the estimation errors that are low.
        idx = err<self.error_threshold
        self.idx_err_lo = np.squeeze(np.where(idx))

        # Create an index of the estimation errors that are high.
        self.idx_err_hi = np.squeeze(np.where(np.logical_not(idx)))

        # Probability of sampling Q-values with high estimation errors.
        # This is either set to the fraction of the replay-memory that
        # has high estimation errors - or it is set to 0.5. So at least
        # half of the batch has high estimation errors.
        prob_err_hi = len(self.idx_err_hi) / self.num_used
        prob_err_hi = max(prob_err_hi, 0.5)

        # Number of samples in a batch that have high estimation errors.
        self.num_samples_err_hi = int(prob_err_hi * batch_size)

        # Number of samples in a batch that have low estimation errors.
        self.num_samples_err_lo = batch_size - self.num_samples_err_hi

    def random_batch(self):
        """
        Get a random batch of states and Q-values from the replay-memory.
        You must call prepare_sampling_prob() before calling this function,
        which also sets the batch-size.

        The batch has been balanced so it contains states and Q-values
        that have both high and low estimation errors for the Q-values.
        This is done to both speed up and stabilize training of the
        Neural Network.
        """

        # Random index of states and Q-values in the replay-memory.
        # These have LOW estimation errors for the Q-values.
        idx_lo = np.random.choice(self.idx_err_lo,
                                  size=self.num_samples_err_lo,
                                  replace=False)

        # Random index of states and Q-values in the replay-memory.
        # These have HIGH estimation errors for the Q-values.
        idx_hi = np.random.choice(self.idx_err_hi,
                                  size=self.num_samples_err_hi,
                                  replace=False)

        # Combine the indices.
        idx = np.concatenate((idx_lo, idx_hi))

        # Get the batches of states and Q-values.
        states_batch = self.states[idx]
        q_values_batch = self.q_values[idx]

        return states_batch, q_values_batch

    def all_batches(self, batch_size=128):
        """
        Iterator for all the states and Q-values in the replay-memory.
        It returns the indices for the beginning and end, as well as
        a progress-counter between 0.0 and 1.0.
        
        This function is not currently being used except by the function
        estimate_all_q_values() below. These two functions are merely
        included to make it easier for you to experiment with the code
        by showing you an easy and efficient way to loop over all the
        data in the replay-memory.
        """

        # Start index for the current batch.
        begin = 0

        # Repeat until all batches have been processed.
        while begin < self.num_used:
            # End index for the current batch.
            end = begin + batch_size

            # Ensure the batch does not exceed the used replay-memory.
            if end > self.num_used:
                end = self.num_used

            # Progress counter.
            progress = end / self.num_used

            # Yield the batch indices and completion-counter.
            yield begin, end, progress

            # Set the start-index for the next batch to the end of this batch.
            begin = end

    def estimate_all_q_values(self, model):
        """
        Estimate all Q-values for the states in the replay-memory
        using the model / Neural Network.

        Note that this function is not currently being used. It is provided
        to make it easier for you to experiment with this code, by showing
        you an efficient way to iterate over all the states and Q-values.

        :param model:
            Instance of the NeuralNetwork-class.
        """

        print("Re-calculating all Q-values in replay memory ...")

        # Process the entire replay-memory in batches.
        for begin, end, progress in self.all_batches():
            # Print progress.
            msg = "\tProgress: {0:.0%}"
            msg = msg.format(progress)
            print_progress(msg)

            # Get the states for the current batch.
            states = self.states[begin:end]

            # Calculate the Q-values using the Neural Network
            # and update the replay-memory.
            self.q_values[begin:end] = model.get_q_values(states=states)

        # Newline.
        print()

    def print_statistics(self):
        """Print statistics for the contents of the replay-memory."""

        print("Replay-memory statistics:")

        # Print statistics for the Q-values before they were updated
        # in update_all_q_values().
        msg = "\tQ-values Before, Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(self.q_values_old),
                         np.mean(self.q_values_old),
                         np.max(self.q_values_old)))

        # Print statistics for the Q-values after they were updated
        # in update_all_q_values().
        msg = "\tQ-values After,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(self.q_values),
                         np.mean(self.q_values),
                         np.max(self.q_values)))

        # Print statistics for the difference in Q-values before and
        # after the update in update_all_q_values().
        q_dif = self.q_values - self.q_values_old
        msg = "\tQ-values Diff.,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(q_dif),
                         np.mean(q_dif),
                         np.max(q_dif)))

        # Print statistics for the number of large estimation errors.
        # Don't use the estimation error for the last state in the memory,
        # because its Q-values have not been updated.
        err = self.estimation_errors[:-1]
        err_count = np.count_nonzero(err > self.error_threshold)
        msg = "\tNumber of large errors > {0}: {1} / {2} ({3:.1%})"
        print(msg.format(self.error_threshold, err_count,
                         self.num_used, err_count / self.num_used))

        # How much of the replay-memory is used by states with end_life.
        end_life_pct = np.count_nonzero(self.end_life) / self.num_used

        # How much of the replay-memory is used by states with end_episode.
        end_episode_pct = np.count_nonzero(self.end_episode) / self.num_used

        # How much of the replay-memory is used by states with non-zero reward.
        reward_nonzero_pct = np.count_nonzero(self.rewards) / self.num_used

        # Print those statistics.
        msg = "\tend_life: {0:.1%}, end_episode: {1:.1%}, reward non-zero: {2:.1%}"
        print(msg.format(end_life_pct, end_episode_pct, reward_nonzero_pct))

########################################################################


class LinearControlSignal:
    """
    A control signal that changes linearly over time.

    This is used to change e.g. the learning-rate for the optimizer
    of the Neural Network, as well as other parameters.
    
    TensorFlow has functionality for doing this, but it uses the
    global_step counter inside the TensorFlow graph, while we
    want the control signals to use a state-counter for the
    game-environment. So it is easier to make this in Python.
    """

    def __init__(self, start_value, end_value, num_iterations, repeat=False):
        """
        Create a new object.

        :param start_value:
            Start-value for the control signal.

        :param end_value:
            End-value for the control signal.

        :param num_iterations:
            Number of iterations it takes to reach the end_value
            from the start_value.

        :param repeat:
            Boolean whether to reset the control signal back to the start_value
            after the end_value has been reached.
        """

        # Store arguments in this object.
        self.start_value = start_value
        self.end_value = end_value
        self.num_iterations = num_iterations
        self.repeat = repeat

        # Calculate the linear coefficient.
        self._coefficient = (end_value - start_value) / num_iterations

    def get_value(self, iteration):
        """Get the value of the control signal for the given iteration."""

        if self.repeat:
            iteration %= self.num_iterations

        if iteration < self.num_iterations:
            value = iteration * self._coefficient + self.start_value
        else:
            value = self.end_value

        return value

########################################################################


class EpsilonGreedy:
    """
    The epsilon-greedy policy either takes a random action with
    probability epsilon, or it takes the action for the highest
    Q-value.
    
    If epsilon is 1.0 then the actions are always random.
    If epsilon is 0.0 then the actions are always argmax for the Q-values.

    Epsilon is typically decreased linearly from 1.0 to 0.1
    and this is also implemented in this class.

    During testing, epsilon is usually chosen lower, e.g. 0.05 or 0.01
    """

    def __init__(self, num_actions,
                 epsilon_testing=0.05,
                 num_iterations=1e6,
                 start_value=1.0, end_value=0.1,
                 repeat=False):
        """
        
        :param num_actions:
            Number of possible actions in the game-environment.

        :param epsilon_testing:
            Epsilon-value when testing.

        :param num_iterations:
            Number of training iterations required to linearly
            decrease epsilon from start_value to end_value.
            
        :param start_value:
            Starting value for linearly decreasing epsilon.

        :param end_value:
            Ending value for linearly decreasing epsilon.

        :param repeat:
            Boolean whether to repeat and restart the linear decrease
            when the end_value is reached, or only do it once and then
            output the end_value forever after.
        """

        # Store parameters.
        self.num_actions = num_actions
        self.epsilon_testing = epsilon_testing

        # Create a control signal for linearly decreasing epsilon.
        self.epsilon_linear = LinearControlSignal(num_iterations=num_iterations,
                                                  start_value=start_value,
                                                  end_value=end_value,
                                                  repeat=repeat)

    def get_epsilon(self, iteration, training):
        """
        Return the epsilon for the given iteration.
        If training==True then epsilon is linearly decreased,
        otherwise epsilon is a fixed number.
        """

        if training:
            epsilon = self.epsilon_linear.get_value(iteration=iteration)
        else:
            epsilon = self.epsilon_testing

        return epsilon

    def get_action(self, q_values, iteration, training):
        """
        Use the epsilon-greedy policy to select an action.
        
        :param q_values:
            These are the Q-values that are estimated by the Neural Network
            for the current state of the game-environment.
         
        :param iteration:
            This is an iteration counter. Here we use the number of states
            that has been processed in the game-environment.

        :param training:
            Boolean whether we are training or testing the
            Reinforcement Learning agent.

        :return:
            action (integer), epsilon (float)
        """

        epsilon = self.get_epsilon(iteration=iteration, training=training)

        # With probability epsilon.
        if np.random.random() < epsilon:
            # Select a random action.
            action = np.random.randint(low=0, high=self.num_actions)
        else:
            # Otherwise select the action that has the highest Q-value.
            action = np.argmax(q_values)

        return action, epsilon

########################################################################


class NeuralNetwork:
    """
    Creates a Neural Network for Reinforcement Learning (Q-Learning).

    Functions are provided for estimating Q-values from states of the
    game-environment, and for optimizing the Neural Network so it becomes
    better at estimating the Q-values.
    """

    def __init__(self, num_actions, replay_memory, use_pretty_tensor=True):
        """
        :param num_actions:
            Number of discrete actions for the game-environment.

        :param replay_memory: 
            Object-instance of the ReplayMemory-class.

        :param use_pretty_tensor:
            Boolean whether to use PrettyTensor (True) which must then be
            installed, or use the tf.layers API (False) which is already
            built into TensorFlow.
        """

        # Whether to use the PrettyTensor API (True) or tf.layers (False).
        self.use_pretty_tensor = use_pretty_tensor

        # Replay-memory used for sampling random batches.
        self.replay_memory = replay_memory

        # Path for saving/restoring checkpoints.
        self.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")

        # Placeholder variable for inputting states into the Neural Network.
        # A state is a multi-dimensional array holding image-frames from
        # the game-environment.
        self.x = tf.placeholder(dtype=tf.float32, shape=[None] + state_shape, name='x')

        # Placeholder variable for inputting the learning-rate to the optimizer.
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

        # Placeholder variable for inputting the target Q-values
        # that we want the Neural Network to be able to estimate.
        self.q_values_new = tf.placeholder(tf.float32,
                                           shape=[None, num_actions],
                                           name='q_values_new')

        # This is a hack that allows us to save/load the counter for
        # the number of states processed in the game-environment.
        # We will keep it as a variable in the TensorFlow-graph
        # even though it will not actually be used by TensorFlow.
        self.count_states = tf.Variable(initial_value=0,
                                        trainable=False, dtype=tf.int64,
                                        name='count_states')

        # Similarly, this is the counter for the number of episodes.
        self.count_episodes = tf.Variable(initial_value=0,
                                          trainable=False, dtype=tf.int64,
                                          name='count_episodes')

        # TensorFlow operation for increasing count_states.
        self.count_states_increase = tf.assign(self.count_states,
                                               self.count_states + 1)

        # TensorFlow operation for increasing count_episodes.
        self.count_episodes_increase = tf.assign(self.count_episodes,
                                                 self.count_episodes + 1)

        # The Neural Network will be constructed in the following.
        # Note that the architecture of this Neural Network is very
        # different from that used in the original DeepMind papers,
        # which was something like this:
        # Input image:      84 x 84 x 4 (4 gray-scale images of 84 x 84 pixels).
        # Conv layer 1:     16 filters 8 x 8, stride 4, relu.
        # Conv layer 2:     32 filters 4 x 4, stride 2, relu.
        # Fully-conn. 1:    256 units, relu. (Sometimes 512 units).
        # Fully-conn. 2:    num-action units, linear.

        # The DeepMind architecture does a very aggressive downsampling of
        # the input images so they are about 10 x 10 pixels after the final
        # convolutional layer. I found that this resulted in significantly
        # distorted Q-values when using the training method further below.
        # The reason DeepMind could get it working was perhaps that they
        # used a very large replay memory (5x as big as here), and a single
        # optimization iteration was performed after each step of the game,
        # and some more tricks.

        # Initializer for the layers in the Neural Network.
        # If you change the architecture of the network, particularly
        # if you add or remove layers, then you may have to change
        # the stddev-parameter here. The initial weights must result
        # in the Neural Network outputting Q-values that are very close
        # to zero - but the network weights must not be too low either
        # because it will make it hard to train the network.
        # You can experiment with values between 1e-2 and 1e-3.
        init = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)

        if self.use_pretty_tensor:
            # This builds the Neural Network using the PrettyTensor API,
            # which is a very elegant builder API, but some people are
            # having problems installing and using it.

            import prettytensor as pt

            # Wrap the input to the Neural Network in a PrettyTensor object.
            x_pretty = pt.wrap(self.x)

            # Create the convolutional Neural Network using Pretty Tensor.
            with pt.defaults_scope(activation_fn=tf.nn.relu):
                self.q_values = x_pretty. \
                    conv2d(kernel=3, depth=16, stride=2, name='layer_conv1', weights=init). \
                    conv2d(kernel=3, depth=32, stride=2, name='layer_conv2', weights=init). \
                    conv2d(kernel=3, depth=64, stride=1, name='layer_conv3', weights=init). \
                    flatten(). \
                    fully_connected(size=1024, name='layer_fc1', weights=init). \
                    fully_connected(size=1024, name='layer_fc2', weights=init). \
                    fully_connected(size=1024, name='layer_fc3', weights=init). \
                    fully_connected(size=1024, name='layer_fc4', weights=init). \
                    fully_connected(size=num_actions, name='layer_fc_out', weights=init,
                                    activation_fn=None)

            # Loss-function which must be optimized. This is the mean-squared
            # error between the Q-values that are output by the Neural Network
            # and the target Q-values.
            self.loss = self.q_values.l2_regression(target=self.q_values_new)
        else:
            # This builds the Neural Network using the tf.layers API,
            # which is very verbose and inelegant, but should work for everyone.

            # Note that the checkpoints for Tutorial #16 which can be
            # downloaded from the internet only support PrettyTensor.
            # Although the Neural Networks appear to be identical when
            # built using the PrettyTensor and tf.layers APIs,
            # they actually create somewhat different TensorFlow graphs
            # where the variables have different names, which means the
            # checkpoints are incompatible for the two builder APIs.

            # Padding used for the convolutional layers.
            padding = 'SAME'

            # Activation function for all convolutional and fully-connected
            # layers, except the last.
            activation = tf.nn.relu

            # Reference to the lastly added layer of the Neural Network.
            # This makes it easy to add or remove layers.
            net = self.x

            # First convolutional layer.
            net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                                   filters=16, kernel_size=3, strides=2,
                                   padding=padding,
                                   kernel_initializer=init, activation=activation)

            # Second convolutional layer.
            net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                                   filters=32, kernel_size=3, strides=2,
                                   padding=padding,
                                   kernel_initializer=init, activation=activation)

            # Third convolutional layer.
            net = tf.layers.conv2d(inputs=net, name='layer_conv3',
                                   filters=64, kernel_size=3, strides=1,
                                   padding=padding,
                                   kernel_initializer=init, activation=activation)

            # Flatten output of the last convolutional layer so it can
            # be input to a fully-connected (aka. dense) layer.
            # TODO: For some bizarre reason, this function is not yet in tf.layers
            # TODO: net = tf.layers.flatten(net)
            net = tf.contrib.layers.flatten(net)

            # First fully-connected (aka. dense) layer.
            net = tf.layers.dense(inputs=net, name='layer_fc1', units=1024,
                                  kernel_initializer=init, activation=activation)

            # Second fully-connected layer.
            net = tf.layers.dense(inputs=net, name='layer_fc2', units=1024,
                                  kernel_initializer=init, activation=activation)

            # Third fully-connected layer.
            net = tf.layers.dense(inputs=net, name='layer_fc3', units=1024,
                                  kernel_initializer=init, activation=activation)

            # Fourth fully-connected layer.
            net = tf.layers.dense(inputs=net, name='layer_fc4', units=1024,
                                  kernel_initializer=init, activation=activation)

            # Final fully-connected layer.
            net = tf.layers.dense(inputs=net, name='layer_fc_out', units=num_actions,
                                  kernel_initializer=init, activation=None)

            # The output of the Neural Network is the estimated Q-values
            # for each possible action in the game-environment.
            self.q_values = net

            # TensorFlow has a built-in loss-function for doing regression:
            # self.loss = tf.nn.l2_loss(self.q_values - self.q_values_new)
            # But it uses tf.reduce_sum() rather than tf.reduce_mean()
            # which is used by PrettyTensor. This means the scale of the
            # gradient is different and hence the hyper-parameters
            # would have to be re-tuned. So instead we calculate the
            # L2-loss similarly to how it is done in PrettyTensor.
            squared_error = tf.square(self.q_values - self.q_values_new)
            sum_squared_error = tf.reduce_sum(squared_error, axis=1)
            self.loss = tf.reduce_mean(sum_squared_error)

        # Optimizer used for minimizing the loss-function.
        # Note the learning-rate is a placeholder variable so we can
        # lower the learning-rate as optimization progresses.
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # Used for saving and loading checkpoints.
        self.saver = tf.train.Saver()

        # Create a new TensorFlow session so we can run the Neural Network.
        self.session = tf.Session()

        # Load the most recent checkpoint if it exists,
        # otherwise initialize all the variables in the TensorFlow graph.
        self.load_checkpoint()

    def close(self):
        """Close the TensorFlow session."""
        self.session.close()

    def load_checkpoint(self):
        """
        Load all variables of the TensorFlow graph from a checkpoint.
        If the checkpoint does not exist, then initialize all variables.
        """

        try:
            print("Trying to restore last checkpoint ...")

            # Use TensorFlow to find the latest checkpoint - if any.
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)

            # Try and load the data in the checkpoint.
            self.saver.restore(self.session, save_path=last_chk_path)

            # If we get to this point, the checkpoint was successfully loaded.
            print("Restored checkpoint from:", last_chk_path)
        except:
            # If the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Failed to restore checkpoint from:", checkpoint_dir)
            print("Initializing variables instead.")
            self.session.run(tf.global_variables_initializer())

    def save_checkpoint(self, current_iteration):
        """Save all variables of the TensorFlow graph to a checkpoint."""

        self.saver.save(self.session,
                        save_path=self.checkpoint_path,
                        global_step=current_iteration)

        print("Saved checkpoint.")

    def get_q_values(self, states):
        """
        Calculate and return the estimated Q-values for the given states.

        A single state contains two images (or channels): The most recent
        image-frame from the game-environment, and a motion-tracing image.
        See the MotionTracer-class for details.

        The input to this function is an array of such states which allows
        for batch-processing of the states. So the input is a 4-dim
        array with shape: [batch, height, width, state_channels].
        
        The output of this function is an array of Q-value-arrays.
        There is a Q-value for each possible action in the game-environment.
        So the output is a 2-dim array with shape: [batch, num_actions]
        """

        # Create a feed-dict for inputting the states to the Neural Network.
        feed_dict = {self.x: states}

        # Use TensorFlow to calculate the estimated Q-values for these states.
        values = self.session.run(self.q_values, feed_dict=feed_dict)

        return values

    def optimize(self, min_epochs=1.0, max_epochs=10,
                 batch_size=128, loss_limit=0.015,
                 learning_rate=1e-3):
        """
        Optimize the Neural Network by sampling states and Q-values
        from the replay-memory.

        The original DeepMind paper performed one optimization iteration
        after processing each new state of the game-environment. This is
        an un-natural way of doing optimization of Neural Networks.

        So instead we perform a full optimization run every time the
        Replay Memory is full (or it is filled to the desired fraction).
        This also gives more efficient use of a GPU for the optimization.

        The problem is that this may over-fit the Neural Network to whatever
        is in the replay-memory. So we use several tricks to try and adapt
        the number of optimization iterations.

        :param min_epochs:
            Minimum number of optimization epochs. One epoch corresponds
            to the replay-memory being used once. However, as the batches
            are sampled randomly and biased somewhat, we may not use the
            whole replay-memory. This number is just a convenient measure.

        :param max_epochs:
            Maximum number of optimization epochs.

        :param batch_size:
            Size of each random batch sampled from the replay-memory.

        :param loss_limit:
            Optimization continues until the average loss-value of the
            last 100 batches is below this value (or max_epochs is reached).

        :param learning_rate:
            Learning-rate to use for the optimizer.
        """

        print("Optimizing Neural Network to better estimate Q-values ...")
        print("\tLearning-rate: {0:.1e}".format(learning_rate))
        print("\tLoss-limit: {0:.3f}".format(loss_limit))
        print("\tMax epochs: {0:.1f}".format(max_epochs))

        # Prepare the probability distribution for sampling the replay-memory.
        self.replay_memory.prepare_sampling_prob(batch_size=batch_size)

        # Number of optimization iterations corresponding to one epoch.
        iterations_per_epoch = self.replay_memory.num_used / batch_size

        # Minimum number of iterations to perform.
        min_iterations = int(iterations_per_epoch * min_epochs)

        # Maximum number of iterations to perform.
        max_iterations = int(iterations_per_epoch * max_epochs)

        # Buffer for storing the loss-values of the most recent batches.
        loss_history = np.zeros(100, dtype=float)

        for i in range(max_iterations):
            # Randomly sample a batch of states and target Q-values
            # from the replay-memory. These are the Q-values that we
            # want the Neural Network to be able to estimate.
            state_batch, q_values_batch = self.replay_memory.random_batch()

            # Create a feed-dict for inputting the data to the TensorFlow graph.
            # Note that the learning-rate is also in this feed-dict.
            feed_dict = {self.x: state_batch,
                         self.q_values_new: q_values_batch,
                         self.learning_rate: learning_rate}

            # Perform one optimization step and get the loss-value.
            loss_val, _ = self.session.run([self.loss, self.optimizer],
                                           feed_dict=feed_dict)

            # Shift the loss-history and assign the new value.
            # This causes the loss-history to only hold the most recent values.
            loss_history = np.roll(loss_history, 1)
            loss_history[0] = loss_val

            # Calculate the average loss for the previous batches.
            loss_mean = np.mean(loss_history)

            # Print status.
            pct_epoch = i / iterations_per_epoch
            msg = "\tIteration: {0} ({1:.2f} epoch), Batch loss: {2:.4f}, Mean loss: {3:.4f}"
            msg = msg.format(i, pct_epoch, loss_val, loss_mean)
            print_progress(msg)

            # Stop the optimization if we have performed the required number
            # of iterations and the loss-value is sufficiently low.
            if i > min_iterations and loss_mean < loss_limit:
                break

        # Print newline.
        print()

    def get_weights_variable(self, layer_name):
        """
        Return the variable inside the TensorFlow graph for the weights
        in the layer with the given name.

        Note that the actual values of the variables are not returned,
        you must use the function get_variable_value() for that.
        """

        if self.use_pretty_tensor:
            # PrettyTensor uses this name for the weights in a conv-layer.
            variable_name = 'weights'
        else:
            # The tf.layers API uses this name for the weights in a conv-layer.
            variable_name = 'kernel'

        with tf.variable_scope(layer_name, reuse=True):
            variable = tf.get_variable(variable_name)

        return variable

    def get_variable_value(self, variable):
        """Return the value of a variable inside the TensorFlow graph."""

        weights = self.session.run(variable)

        return weights

    def get_layer_tensor(self, layer_name):
        """
        Return the tensor for the output of a layer.
        Note that this does not return the actual values,
        but instead returns a reference to the tensor
        inside the TensorFlow graph. Use get_tensor_value()
        to get the actual contents of the tensor.
        """

        # The name of the last operation of a layer,
        # assuming it uses Relu as the activation-function.
        tensor_name = layer_name + "/Relu:0"

        # Get the tensor with this name.
        tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

        return tensor

    def get_tensor_value(self, tensor, state):
        """Get the value of a tensor in the Neural Network."""

        # Create a feed-dict for inputting the state to the Neural Network.
        feed_dict = {self.x: [state]}

        # Run the TensorFlow session to calculate the value of the tensor.
        output = self.session.run(tensor, feed_dict=feed_dict)

        return output

    def get_count_states(self):
        """
        Get the number of states that has been processed in the game-environment.
        This is not used by the TensorFlow graph. It is just a hack to save and
        reload the counter along with the checkpoint-file.
        """
        return self.session.run(self.count_states)

    def get_count_episodes(self):
        """
        Get the number of episodes that has been processed in the game-environment.
        """
        return self.session.run(self.count_episodes)

    def increase_count_states(self):
        """
        Increase the number of states that has been processed
        in the game-environment.
        """
        return self.session.run(self.count_states_increase)

    def increase_count_episodes(self):
        """
        Increase the number of episodes that has been processed
        in the game-environment.
        """
        return self.session.run(self.count_episodes_increase)

########################################################################


class Agent:
    """
    This implements the function for running the game-environment with
    an agent that uses Reinforcement Learning. This class also creates
    instances of the Replay Memory and Neural Network.
    """

    def __init__(self, env_name, training, render=False, use_logging=True):
        """
        Create an object-instance. This also creates a new object for the
        Replay Memory and the Neural Network.
        
        Replay Memory will only be allocated if training==True.

        :param env_name:
            Name of the game-environment in OpenAI Gym.
            Examples: 'Breakout-v0' and 'SpaceInvaders-v0'

        :param training:
            Boolean whether to train the agent and Neural Network (True),
            or test the agent by playing a number of episodes of the game (False).
        
        :param render:
            Boolean whether to render the game-images to screen during testing.

        :param use_logging:
            Boolean whether to use logging to text-files during training.
        """

        # Create the game-environment using OpenAI Gym.
        self.env = gym.make(env_name)

        # The number of possible actions that the agent may take in every step.
        self.num_actions = self.env.action_space.n

        # Whether we are training (True) or testing (False).
        self.training = training

        # Whether to render each image-frame of the game-environment to screen.
        self.render = render

        # Whether to use logging during training.
        self.use_logging = use_logging

        if self.use_logging and self.training:
            # Used for logging Q-values and rewards during training.
            self.log_q_values = LogQValues()
            self.log_reward = LogReward()
        else:
            self.log_q_values = None
            self.log_reward = None

        # List of string-names for the actions in the game-environment.
        self.action_names = self.env.unwrapped.get_action_meanings()

        # Epsilon-greedy policy for selecting an action from the Q-values.
        # During training the epsilon is decreased linearly over the given
        # number of iterations. During testing the fixed epsilon is used.
        self.epsilon_greedy = EpsilonGreedy(start_value=1.0,
                                            end_value=0.1,
                                            num_iterations=1e6,
                                            num_actions=self.num_actions,
                                            epsilon_testing=0.01)

        if self.training:
            # The following control-signals are only used during training.

            # The learning-rate for the optimizer decreases linearly.
            self.learning_rate_control = LinearControlSignal(start_value=1e-3,
                                                             end_value=1e-5,
                                                             num_iterations=5e6)

            # The loss-limit is used to abort the optimization whenever the
            # mean batch-loss falls below this limit.
            self.loss_limit_control = LinearControlSignal(start_value=0.1,
                                                          end_value=0.015,
                                                          num_iterations=5e6)

            # The maximum number of epochs to perform during optimization.
            # This is increased from 5 to 10 epochs, because it was found for
            # the Breakout-game that too many epochs could be harmful early
            # in the training, as it might cause over-fitting.
            # Later in the training we would occasionally get rare events
            # and would therefore have to optimize for more iterations
            # because the learning-rate had been decreased.
            self.max_epochs_control = LinearControlSignal(start_value=5.0,
                                                          end_value=10.0,
                                                          num_iterations=5e6)

            # The fraction of the replay-memory to be used.
            # Early in the training, we want to optimize more frequently
            # so the Neural Network is trained faster and the Q-values
            # are learned and updated more often. Later in the training,
            # we need more samples in the replay-memory to have sufficient
            # diversity, otherwise the Neural Network will over-fit.
            self.replay_fraction = LinearControlSignal(start_value=0.1,
                                                       end_value=1.0,
                                                       num_iterations=5e6)
        else:
            # We set these objects to None when they will not be used.
            self.learning_rate_control = None
            self.loss_limit_control = None
            self.max_epochs_control = None
            self.replay_fraction = None

        if self.training:
            # We only create the replay-memory when we are training the agent,
            # because it requires a lot of RAM. The image-frames from the
            # game-environment are resized to 105 x 80 pixels gray-scale,
            # and each state has 2 channels (one for the recent image-frame
            # of the game-environment, and one for the motion-trace).
            # Each pixel is 1 byte, so this replay-memory needs more than
            # 3 GB RAM (105 x 80 x 2 x 200000 bytes).

            self.replay_memory = ReplayMemory(size=200000,
                                              num_actions=self.num_actions)
        else:
            self.replay_memory = None

        # Create the Neural Network used for estimating Q-values.
        self.model = NeuralNetwork(num_actions=self.num_actions,
                                   replay_memory=self.replay_memory)

        # Log of the rewards obtained in each episode during calls to run()
        self.episode_rewards = []

    def reset_episode_rewards(self):
        """Reset the log of episode-rewards."""
        self.episode_rewards = []

    def get_action_name(self, action):
        """Return the name of an action."""
        return self.action_names[action]

    def get_lives(self):
        """Get the number of lives the agent has in the game-environment."""
        return self.env.unwrapped.ale.lives()

    def run(self, num_episodes=None):
        """
        Run the game-environment and use the Neural Network to decide
        which actions to take in each step through Q-value estimates.
        
        :param num_episodes: 
            Number of episodes to process in the game-environment.
            If None then continue forever. This is useful during training
            where you might want to stop the training using Ctrl-C instead.
        """

        # This will cause a reset in the first iteration of the following loop.
        end_episode = True

        # Counter for the number of states we have processed.
        # This is stored in the TensorFlow graph so it can be
        # saved and reloaded along with the checkpoint.
        count_states = self.model.get_count_states()

        # Counter for the number of episodes we have processed.
        count_episodes = self.model.get_count_episodes()

        if num_episodes is None:
            # Loop forever by comparing the episode-counter to infinity.
            num_episodes = float('inf')
        else:
            # The episode-counter may not start at zero if training is
            # continued from a checkpoint. Take this into account
            # when determining the number of iterations to perform.
            num_episodes += count_episodes

        while count_episodes <= num_episodes:
            if end_episode:
                # Reset the game-environment and get the first image-frame.
                img = self.env.reset()

                # Create a new motion-tracer for processing images from the
                # game-environment. Initialize with the first image-frame.
                # This resets the motion-tracer so the trace starts again.
                # This could also be done if end_life==True.
                motion_tracer = MotionTracer(img)

                # Reset the reward for the entire episode to zero.
                # This is only used for printing statistics.
                reward_episode = 0.0

                # Increase the counter for the number of episodes.
                # This counter is stored inside the TensorFlow graph
                # so it can be saved and restored with the checkpoint.
                count_episodes = self.model.increase_count_episodes()

                # Get the number of lives that the agent has left in this episode.
                num_lives = self.get_lives()

            # Get the state of the game-environment from the motion-tracer.
            # The state has two images: (1) The last image-frame from the game
            # and (2) a motion-trace that shows movement trajectories.
            state = motion_tracer.get_state()

            # Use the Neural Network to estimate the Q-values for the state.
            # Note that the function assumes an array of states and returns
            # a 2-dim array of Q-values, but we just have a single state here.
            q_values = self.model.get_q_values(states=[state])[0]

            # Determine the action that the agent must take in the game-environment.
            # The epsilon is just used for printing further below.
            action, epsilon = self.epsilon_greedy.get_action(q_values=q_values,
                                                             iteration=count_states,
                                                             training=self.training)

            # Take a step in the game-environment using the given action.
            # Note that in OpenAI Gym, the step-function actually repeats the
            # action between 2 and 4 time-steps for Atari games, with the number
            # chosen at random.
            img, reward, end_episode, info = self.env.step(action=action)

            # Process the image from the game-environment in the motion-tracer.
            # This will first be used in the next iteration of the loop.
            motion_tracer.process(image=img)

            # Add the reward for the step to the reward for the entire episode.
            reward_episode += reward

            # Determine if a life was lost in this step.
            num_lives_new = self.get_lives()
            end_life = (num_lives_new < num_lives)
            num_lives = num_lives_new

            # Increase the counter for the number of states that have been processed.
            count_states = self.model.increase_count_states()

            if not self.training and self.render:
                # Render the game-environment to screen.
                self.env.render()

                # Insert a small pause to slow down the game,
                # making it easier to follow for human eyes.
                time.sleep(0.01)

            # If we want to train the Neural Network to better estimate Q-values.
            if self.training:
                # Add the state of the game-environment to the replay-memory.
                self.replay_memory.add(state=state,
                                       q_values=q_values,
                                       action=action,
                                       reward=reward,
                                       end_life=end_life,
                                       end_episode=end_episode)

                # How much of the replay-memory should be used.
                use_fraction = self.replay_fraction.get_value(iteration=count_states)

                # When the replay-memory is sufficiently full.
                if self.replay_memory.is_full() \
                    or self.replay_memory.used_fraction() > use_fraction:

                    # Update all Q-values in the replay-memory through a backwards-sweep.
                    self.replay_memory.update_all_q_values()

                    # Log statistics for the Q-values to file.
                    if self.use_logging:
                        self.log_q_values.write(count_episodes=count_episodes,
                                                count_states=count_states,
                                                q_values=self.replay_memory.q_values)

                    # Get the control parameters for optimization of the Neural Network.
                    # These are changed linearly depending on the state-counter.
                    learning_rate = self.learning_rate_control.get_value(iteration=count_states)
                    loss_limit = self.loss_limit_control.get_value(iteration=count_states)
                    max_epochs = self.max_epochs_control.get_value(iteration=count_states)

                    # Perform an optimization run on the Neural Network so as to
                    # improve the estimates for the Q-values.
                    # This will sample random batches from the replay-memory.
                    self.model.optimize(learning_rate=learning_rate,
                                        loss_limit=loss_limit,
                                        max_epochs=max_epochs)

                    # Save a checkpoint of the Neural Network so we can reload it.
                    self.model.save_checkpoint(count_states)

                    # Reset the replay-memory. This throws away all the data we have
                    # just gathered, so we will have to fill the replay-memory again.
                    self.replay_memory.reset()

            if end_episode:
                # Add the episode's reward to a list for calculating statistics.
                self.episode_rewards.append(reward_episode)

            # Mean reward of the last 30 episodes.
            if len(self.episode_rewards) == 0:
                # The list of rewards is empty.
                reward_mean = 0.0
            else:
                reward_mean = np.mean(self.episode_rewards[-30:])

            if self.training and end_episode:
                # Log reward to file.
                if self.use_logging:
                    self.log_reward.write(count_episodes=count_episodes,
                                          count_states=count_states,
                                          reward_episode=reward_episode,
                                          reward_mean=reward_mean)

                # Print reward to screen.
                msg = "{0:4}:{1}\t Epsilon: {2:4.2f}\t Reward: {3:.1f}\t Episode Mean: {4:.1f}"
                print(msg.format(count_episodes, count_states, epsilon,
                                 reward_episode, reward_mean))
            elif not self.training and (reward != 0.0 or end_life or end_episode):
                # Print Q-values and reward to screen.
                msg = "{0:4}:{1}\tQ-min: {2:5.3f}\tQ-max: {3:5.3f}\tLives: {4}\tReward: {5:.1f}\tEpisode Mean: {6:.1f}"
                print(msg.format(count_episodes, count_states, np.min(q_values),
                                 np.max(q_values), num_lives, reward_episode, reward_mean))

########################################################################

if __name__ == '__main__':
    # Description of this program.
    desc = "Reinformenct Learning (Q-learning) for Atari Games using TensorFlow."

    # Create the argument parser.
    parser = argparse.ArgumentParser(description=desc)

    # Add arguments to the parser.
    parser.add_argument("--env", required=False, default='Breakout-v0',
                        help="name of the game-environment in OpenAI Gym")

    parser.add_argument("--training", required=False,
                        dest='training', action='store_true',
                        help="train the agent (otherwise test the agent)")

    parser.add_argument("--render", required=False,
                        dest='render', action='store_true',
                        help="render game-output to screen")

    parser.add_argument("--episodes", required=False, type=int, default=None,
                        help="number of episodes to run")

    parser.add_argument("--dir", required=False, default=checkpoint_base_dir,
                        help="directory for the checkpoint and log-files")

    # Parse the command-line arguments.
    args = parser.parse_args()

    # Get the arguments.
    env_name = args.env
    training = args.training
    render = args.render
    num_episodes = args.episodes
    checkpoint_base_dir = args.dir

    # Update all the file-paths after the base-dir has been set.
    update_paths(env_name=env_name)

    # Create an agent for either training or testing on the game-environment.
    agent = Agent(env_name=env_name,
                  training=training,
                  render=render)

    # Run the agent
    agent.run(num_episodes=num_episodes)

    # Print statistics.
    rewards = agent.episode_rewards
    print()  # Newline.
    print("Rewards for {0} episodes:".format(len(rewards)))
    print("- Min:   ", np.min(rewards))
    print("- Mean:  ", np.mean(rewards))
    print("- Max:   ", np.max(rewards))
    print("- Stdev: ", np.std(rewards))

########################################################################
