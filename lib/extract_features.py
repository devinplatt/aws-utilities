# This module contains functions to extract mel spectrum features from audio
# files.
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
#   audio: audio as a numpy array.
# Output:
#   A tuple containing the log amplitude mel spectrum features, as well as
#   metadata regarding the parameters of the transformation, in the format:
#       (numpy array of shape (frames, mel bands), parameter dictionary)
#
# Terminology:
#   frame:  a frame essentially our "atomic unit". It is the output of the log
#           mel spectrum transformation before we do any stacking (not done in
#           this function). The length of a frame is dependent on the sample
#           rate and hop length, and can be computed as:
#
#               frame duration = hop / sr
#
# The default frame duration is 23 ms.
@timeit
def transform_audio(audio,
                    n_fft=2048,
                    n_mels=40,
                    sr=22050,
                    hop_length=512,
                    fmin=None,
                    fmax=None):
    # Midi values of 24 (C2) and 120 (C10) are chosen, since humans typically
    # can't hear much beyond this range.
    if not fmin:
        fmin = librosa.midi_to_hz(24)
    if not fmax:
        fmax = librosa.midi_to_hz(120)
    # First stage is a mel-frequency specrogram of bounded range.
    mel = librosa.feature.melspectrogram(audio,
                                         sr=sr,
                                         n_fft=n_fft,
                                         hop_length=hop_length,
                                         n_mels=n_mels,
                                         fmax=fmax,
                                         fmin=fmin)
    # Second stage is log-amplitude; power is relative to peak in the signal.
    log_amplitude = librosa.logamplitude(mel, ref_power=np.max)
    # Third stage transposes the data so that frames become samples.
    # Its shape is:
    # (length of audio / frame duration, number of mel bands)
    transpose = np.transpose(log_amplitude)
    return (transpose,
            {'n_fft': n_fft, 'n_mels': n_mels, 'sr': sr,
            'hop_length': hop_length, 'fmin': fmin, 'fmax': fmax})


# Take an input mp3 file name, and extract mel features, saving the resulting
# feature file.
def extract_one(input_mp3_file_name, output_feature_file_name, **kwargs):
    audio, _ = librosa.load(input_mp3_file_name)
    features = transform_audio(audio, **kwargs)
    # Using compress=1 to make sure it is stored as one file.
    joblib.dump(features, output_feature_file_name, compress=1)


def try_extract_one(input_mp3_file_name, output_feature_file_name, **kwargs):
    try:
        extract_one(input_mp3_file_name, output_feature_file_name, **kwargs)
    except Exception as e:
        print(e)
        return False
    else:
        return True
