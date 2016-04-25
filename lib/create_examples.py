# This module contains functions to join/split mel spectrum features into
# input examples for a model.

import numpy as np
import librosa
import sklearn
import sklearn.cluster
import sklearn.pipeline
import csv
from collections import defaultdict
import random
import joblib
import os

from extraction_worker.lib.core import timeit


# Extract log amplitude mel spectrum features from audio data.
# Input:
#   mel: Log amplitude mel spectrum features, and parameters regarding how
#         the features were extracted, in the format as created in
#         extraction_worker/lib/extract_features.py:
#             (numpy array of shape (frames, mel bands), parameter dictionary)
#   labels: the labels of the data.
#   approximate_window_length_in_ms: the desired length of training examples.
# Output:
#   X:  Log amplitude mel spectrum features. This function computes feature
#       examples for every half second of audio.
#   y:  an expanded label vector (since there will be multiple feature examples
#       for any input longer than half a second.)
#   parameters:
#       the parameter dictionary from the mel input, but with an added
#       approximate_window_length_in_ms value.
#
# Terminology:
#   record: a record is a full track (a record is one item of input to this
#           function)
#   window: a window is the length of an output feature example.
#   frame:  a frame essentially our "atomic unit". It is the output of the log
#           mel spectrum transformation before we do any stacking.
#           The length of a frame is dependent on the sample rate and hop
#           length, and can be computed as:
#
#               frame duration = hop / sr
%timeit
def mel_to_example(mel, label,
                   approximate_window_length_in_ms=500,
                   verbose=False):
    parameters = mel[1]
    parameters['approximate_window_length_in_ms'] = approximate_window_length_in_ms
    frame_duration = parameters['hop_length'] / float(parameters['sr'])
    number_frames_per_window = int(approximate_window_length_in_ms / (1000 * frame_duration))
    number_frames = len(mel[0])
    # Number of windows is the same as the number of examples.
    num_windows_per_record = int(number_frames / number_frames_per_window)
    X_combined = [np.hstack(
            mel[0][number_frames_per_window * i:
                   number_frames_per_window * (i + 1)]) for i in range(num_windows_per_record)]
    # We duplicated the label for each example.
    y_scaled = [label for _ in range(num_windows_per_record)]
    # We save the stackpipe for training.
    # Last stage stacks all samples together into one matrix for training.
    # Stack = librosa.util.FeatureExtractor(np.vstack, iterate=False)
    # stackpipe = sklearn.pipeline.Pipeline([('Stack', Stack)])
    # X_scaled = stackpipe.transform(X_combined)
    if verbose:
        print('Number of total sample frames: {}'.format(number_frames))
        print('Number of frames per example: {}'.format(number_frames_per_window))
        print('Number of examples: {}'.format(num_windows_per_record))
    return X_combined, y_scaled, parameters


# Take an input mp3 file name, and extract mel features, saving the resulting
# feature file.
def extract_one(input_mel_file_name, output_examples_file_name):
    # Load audio.
    mel = joblib.load(input_mel_file_name)
    examples, labels, parameters = mel_to_example(mel, labels)
    # Using compress=1 to make sure it is stored as one file.
    joblib.dump((examples, labels, parameters), output_feature_file_name,
                compress=1)


def try_extract_one(input_mp3_file_name, output_feature_file_name):
    try:
        extract_one(input_mp3_file_name, output_feature_file_name)
    except Exception as e:
        print(e)
        return False
    else:
        return True
