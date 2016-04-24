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

from aws_utilities.utils.core_utils import timeit

# We'll build the feature pipeline object here
# This is based on:
# http://nbviewer.ipython.org/github/bmcfee/librosa/blob/master/examples/LibROSA%20sklearn%20feature%20pipeline.ipynb

# First stage is a mel-frequency specrogram of bounded range.
MelSpec = librosa.util.FeatureExtractor(librosa.feature.melspectrogram, 
                                        n_fft=2048,
                                        n_mels=40,  # 128,
                                        fmax=librosa.midi_to_hz(116), 
                                        fmin=librosa.midi_to_hz(24))

# Second stage is log-amplitude; power is relative to peak in the signal.
LogAmp = librosa.util.FeatureExtractor(librosa.logamplitude, 
                                       ref_power=np.max)


# Third stage transposes the data so that frames become samples.
Transpose = librosa.util.FeatureExtractor(np.transpose)

# Last stage stacks all samples together into one matrix for training.
Stack = librosa.util.FeatureExtractor(np.vstack, iterate=False)

# Extract log amplitude mel spectrum features from audio data.
# Input:
#   data: a list of audio data as numpy arrays.
#   labels: the labels of the data.
# Output:
#   X:  Log amplitude mel spectrum features. This function computes feature
#       examples for every half second of audio.
#   y:  an expanded label vector (since there will be multiple feature examples
#       for any input longer than half a second.)
#
# Terminology:
#   record: a record is a full track (a record is one item of input to this
#           function)
#   window: a window is the length of an output feature example.
#   frame:  a frame essentially our "atomic unit". It is the output of the log
#           mel spectrum transformation before we do any stacking.
#           The length of a frame is dependent on the sample rate and hop
#           length, and can be computed as:
#           frame duration = hop / sr
#
# At the moment the frame duration is 23 ms. The window length is 487 ms (half
# a second).
#
# Note that at the moment we are assuming all input audio is of the same length.
# TODO: amend this to accept variable length input.
# TODO: remove hard-coded values (eg. 21). Improve documentation explanation.
@timeit
def transform_data(data, labels, verbose=False):
    melpipe = sklearn.pipeline.Pipeline([('Mel spectrogram', MelSpec),
                                         ('Log amplitude', LogAmp)])
    X_mel = melpipe.transform(data)
    transpipe = sklearn.pipeline.Pipeline([('Transpose', Transpose)])
    X_trans = transpipe.transform(X_mel)
    # sr = 22050. Hop length = 512 (the default, see the librosa documentation:
    # https://bmcfee.github.io/librosa/generated/librosa.feature.melspectrogram.html)
    # So sample length = hop / sr = ~23 ms. so 21*46ms = .487 s (about half a second)
    num_windows_per_record = int(len(X_trans[0]) / 21)  # 431 / 21 ~= 20. IS THIS ACTUALLY NUMBER WINDOWS PER RECORD?
    # We hstack, so there are 40 mel features, followed by 40 mel features,
    # followed by ... 20 times
    # this happens for each record. So there will be (20 * num_records) examples.
    X_combined = [np.hstack(x[21*i:21*i+21]) for x in X_trans for i in range(num_windows_per_record)]
    stackpipe = sklearn.pipeline.Pipeline([('Stack', Stack)])
    X_scaled = stackpipe.transform(X_combined)
    num_examples_per_record = int(len(X_scaled) / len(X_mel))
    y_scaled = [val for val in labels for _ in range(num_examples_per_record)]
    if verbose:
        print('Number of records: {}'.format(len(X_mel)))
        print('Number of sample frames per record: {}'.format(len(X_trans[0])))
        print('Number of sample windows per record: {}'.format(num_windows_per_record))
        print('Number of examples per record: {}'.format(num_examples_per_record))
        print('Number of examples: {}'.format(len(X_scaled)))
    return X_scaled, y_scaled


# Take an input mp3 file name, and extract mel features, saving the resulting
# feature file.
def extract_one(input_mp3_file_name, output_feature_file_name):
    # Load audio.
    audio, _ = librosa.load(input_mp3_file_name)
    labels = ['label']
    features, _ = transform_data([audio], labels)
    features = features[0]
    # Using compress=1 to make sure it is stored as one file.
    joblib.dump(features, output_feature_file_name, compress=1)
