#!/usr/bin/python

########################################################################
#
# Function and script for converting videos to images.
#
# This can be run as a script in a Linux shell by typing:
#
#    python convert.py
#
# Or by running:
#
#    chmod +x convert.py
#    ./convert.py
#
# Requires the program avconv to be installed.
# Tested with avconv v. 9.18-6 on Linux Mint.
#
# Implemented in Python 3.5 (seems to work in Python 2.7 as well)
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

import os
import subprocess
import argparse

########################################################################


def video2images(in_dir, out_dir, crop_size, out_size, framerate, video_exts):
    """
    Convert videos to images. The videos are located in the directory in_dir
    and all its sub-directories which are processed recursively. The directory
    structure is replicated to out_dir where the jpeg-images are saved.

    :param in_dir:
        Input directory for the videos e.g. "/home/magnus/video/"
        All sub-directories are processed recursively.

    :param out_dir:
        Output directory for the images e.g. "/home/magnus/video-images/"

    :param crop_size:
        Integer. First the videos are cropped to this width and height.

    :param out_size:
        Integer. After cropping, the videos are resized to this width and height.

    :param framerate:
        Integer. Number of frames to grab per second.

    :param video_exts:
        Tuple of strings. Extensions for video-files e.g. ('.mts', '.mp4')
        Not case-sensitive.

    :return:
        Nothing.
    """

    # Convert all video extensions to lower-case.
    video_exts = tuple(ext.lower() for ext in video_exts)

    # Number of videos processed.
    video_count = 0

    # Process all the sub-dirs recursively.
    for current_dir, dir_names, file_names in os.walk(in_dir):
        # The current dir relative to the input directory.
        relative_path = os.path.relpath(current_dir, in_dir)

        # Name of the new directory for the output images.
        new_dir = os.path.join(out_dir, relative_path)

        # If the output-directory does not exist, then create it.
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        # For all the files in the current directory.
        for file_name in file_names:
            # If the file has a valid video-extension. Compare lower-cases.
            if file_name.lower().endswith(video_exts):
                # File-path for the input video.
                in_file = os.path.join(current_dir, file_name)

                # Split the file-path in root and extension.
                file_root, file_ext = os.path.splitext(file_name)

                # Create the template file-name for the output images.
                new_file_name = file_root + "-%4d.jpg"

                # Complete file-path for the output images incl. all sub-dirs.
                new_file_path = os.path.join(new_dir, new_file_name)

                # Clean up the path by removing e.g. "/./"
                new_file_path = os.path.normpath(new_file_path)

                # Print status.
                print("Converting video to images:")
                print("- Input video:   {0}".format(in_file))
                print("- Output images: {0}".format(new_file_path))

                # Command to be run in the shell for the video-conversion tool.
                cmd = "avconv -i {0} -r {1} -vf crop={2}:{2} -vf scale={3}:{3} -qscale 2 {4}"

                # Fill in the arguments for the command-line.
                cmd = cmd.format(in_file, framerate, crop_size, out_size, new_file_path)

                # Run the command-line in a shell.
                subprocess.call(cmd, shell=True)

                # Increase the number of videos processed.
                video_count += 1

                # Print newline.
                print()

    print("Number of videos converted: {0}".format(video_count))


########################################################################
# This script allows you to run the video-conversion from the command-line.

if __name__ == "__main__":
    # Argument description.
    desc = "Convert videos to images. " \
           "Recursively processes all sub-dirs of INDIR " \
           "and replicates the dir-structure to OUTDIR. " \
           "The video is first cropped to CROP:CROP pixels, " \
           "then resized to SIZE:SIZE pixels and written as a jpeg-file. "

    # Create the argument parser.
    parser = argparse.ArgumentParser(description=desc)

    # Add arguments to the parser.
    parser.add_argument("--indir", required=True,
                        help="input directory where videos are located")

    parser.add_argument("--outdir", required=True,
                        help="output directory where images will be saved")

    parser.add_argument("--crop", required=True, type=int,
                        help="the input videos are first cropped to CROP:CROP pixels")

    parser.add_argument("--size", required=True, type=int,
                        help="the input videos are then resized to SIZE:SIZE pixels")

    parser.add_argument("--rate", required=False, type=int, default=5,
                        help="the number of frames to convert per second")

    parser.add_argument("--exts", required=False, nargs="+",
                        help="list of extensions for video-files e.g. .mts .mp4")

    # Parse the command-line arguments.
    args = parser.parse_args()

    # Get the arguments.
    in_dir = args.indir
    out_dir = args.outdir
    crop_size = args.crop
    out_size = args.size
    framerate = args.rate
    video_exts = args.exts

    if video_exts is None:
        # Default extensions for video-files.
        video_exts = (".MTS", ".mp4")
    else:
        # A list of strings is provided as a command-line argument, but we
        # need a tuple instead of a list, so convert it to a tuple.
        video_exts = tuple(video_exts)

    # Print the arguments.
    print("Convert videos to images.")
    print("- Input dir: " + in_dir)
    print("- Output dir: " + out_dir)
    print("- Crop width and height: {0}".format(crop_size))
    print("- Resize width and height: {0}".format(out_size))
    print("- Frame-rate: {0}".format(framerate))
    print("- Video extensions: {0}".format(video_exts))
    print()

    # Perform the conversions.
    video2images(in_dir=in_dir, out_dir=out_dir,
                 crop_size=crop_size, out_size=out_size,
                 framerate=framerate, video_exts=video_exts)

########################################################################
