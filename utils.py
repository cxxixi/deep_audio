import os
import argparse
import numpy as np
import random
import pandas as pd
import time
import librosa
from scipy.stats import skew
import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser()

# General settings
# parser.add_argument('--dataset', required=True, choices=['esc10', 'esc50', 'urbansound8k'])
# parser.add_argument('--netType', required=True, choices=['envnet', 'envnetv2'])
# parser.add_argument('--data', required=True, help='Path to dataset')


parser.add_argument('--dataset', default="esc50", choices=['esc10', 'esc50', 'urbansound8k'])
parser.add_argument('--netType', default="envnet", choices=['envnet', 'envnetv2'])
parser.add_argument('--data', default="./datasets/", help='Path to dataset')
parser.add_argument('--split', type=int, default=-1, help='esc: 1-5, urbansound: 1-10 (-1: run on all splits)')
parser.add_argument('--save', default='None', help='Directory to save the results')
parser.add_argument('--testOnly', action='store_true')
parser.add_argument('--gpu', type=int, default=0)

# Learning settings (default settings are defined below)
parser.add_argument('--BC', action='store_true', help='BC learning')
parser.add_argument('--strongAugment', action='store_true', help='Add scale and gain augmentation')
parser.add_argument('--nEpochs', type=int, default=-1)
parser.add_argument('--LR', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--schedule', type=float, nargs='*', default=-1, help='When to divide the LR')
parser.add_argument('--warmup', type=int, default=-1, help='Number of epochs to warm up')
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--weightDecay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nCrops', type=int, default=3)

args = parser.parse_args()

# Dataset details
# if opt.dataset == 'esc50':
#     opt.nClasses = 50
#     opt.nFolds = 5
# elif opt.dataset == 'esc10':
#     opt.nClasses = 10
#     opt.nFolds = 5
# else:  # urbansound8k
#     opt.nClasses = 10
#     opt.nFolds = 10

# if opt.split == -1:
#     opt.splits = range(1, opt.nFolds + 1)
# else:
#     opt.splits = [opt.split]

# Model details
if args.netType == 'envnet':
    args.fs = 16000
    args.inputLength = 24014
else:  # envnetv2
    args.fs = 44100
    args.inputLength = 66650

# # Default settings (nEpochs will be doubled if opt.BC)
# default_settings = dict()
# default_settings['esc50'] = {
#     'envnet': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.5, 0.75], 'warmup': 0},
#     'envnetv2': {'nEpochs': 1000, 'LR': 0.1, 'schedule': [0.3, 0.6, 0.9], 'warmup': 10}
# }
# default_settings['esc10'] = {
#     'envnet': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.5, 0.75], 'warmup': 0},
#     'envnetv2': {'nEpochs': 600, 'LR': 0.01, 'schedule': [0.5, 0.75], 'warmup': 0}
# }
# default_settings['urbansound8k'] = {
#     'envnet': {'nEpochs': 400, 'LR': 0.01, 'schedule': [0.5, 0.75], 'warmup': 0},
#     'envnetv2': {'nEpochs': 600, 'LR': 0.1, 'schedule': [0.3, 0.6, 0.9], 'warmup': 10}
# }
# for key in ['nEpochs', 'LR', 'schedule', 'warmup']:
#     if eval('opt.{}'.format(key)) == -1:
#         setattr(opt, key, default_settings[opt.dataset][opt.netType][key])
#         if key == 'nEpochs' and opt.BC:
#             opt.nEpochs *= 2

# if opt.save != 'None' and not os.path.isdir(opt.save):
#     os.makedirs(opt.save)

# display_info(opt)

# return opt

# import chainer.functions as F


# Default data augmentation
def padding(pad):
    def f(sound):
        return np.pad(sound, pad, 'constant')

    return f


def random_crop(size):
    def f(sound):
        org_size = len(sound)
        start = random.randint(0, org_size - size)
        return sound[start: start + size]

    return f


def normalize(factor):
    def f(sound):
        return sound / factor

    return f


# For strong data augmentation
def random_scale(max_scale, interpolate='Linear'):
    def f(sound):
        scale = np.power(max_scale, random.uniform(-1, 1))
        output_size = int(len(sound) * scale)
        ref = np.arange(output_size) / scale
        if interpolate == 'Linear':
            ref1 = ref.astype(np.int32)
            ref2 = np.minimum(ref1 + 1, len(sound) - 1)
            r = ref - ref1
            scaled_sound = sound[ref1] * (1 - r) + sound[ref2] * r
        elif interpolate == 'Nearest':
            scaled_sound = sound[ref.astype(np.int32)]
        else:
            raise Exception('Invalid interpolation mode {}'.format(interpolate))

        return scaled_sound

    return f


def random_gain(db):
    def f(sound):
        return sound * np.power(10, random.uniform(-db, db) / 20.0)

    return f


# For testing phase
def multi_crop(input_length, n_crops):
    def f(sound):
        stride = (len(sound) - input_length) // (n_crops - 1)
        sounds = [sound[stride * i: stride * i + input_length] for i in range(n_crops)]
        return np.array(sounds)

    return f


# For BC learning
def a_weight(fs, n_fft, min_db=-80.0):
    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = np.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (2 * np.log10(12194) + 2 * np.log10(freq_sq)
                           - np.log10(freq_sq + 12194 ** 2)
                           - np.log10(freq_sq + 20.6 ** 2)
                           - 0.5 * np.log10(freq_sq + 107.7 ** 2)
                           - 0.5 * np.log10(freq_sq + 737.9 ** 2))
    weight = np.maximum(weight, min_db)

    return weight


def compute_gain(sound, fs, min_db=-80.0, mode='A_weighting'):
    if fs == 16000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    else:
        raise Exception('Invalid fs {}'.format(fs))
    stride = n_fft // 2

    gain = []
    for i in xrange(0, len(sound) - n_fft + 1, stride):
        if mode == 'RMSE':
            g = np.mean(sound[i: i + n_fft] ** 2)
        elif mode == 'A_weighting':
            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])
            power_spec = np.abs(spec) ** 2
            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
            g = np.sum(a_weighted_spec)
        else:
            raise Exception('Invalid mode {}'.format(mode))
        gain.append(g)

    gain = np.array(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)

    return gain_db


def mix(sound1, sound2, r, fs):
    gain1 = np.max(compute_gain(sound1, fs))  # Decibel
    gain2 = np.max(compute_gain(sound2, fs))
    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
    sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))

    return sound


def kl_divergence(y, t):
    entropy = - F.sum(t[t.data.nonzero()] * F.log(t[t.data.nonzero()]))
    crossEntropy = - F.sum(t * F.log_softmax(y))

    return (crossEntropy - entropy) / y.shape[0]


# Convert time representation
def to_hms(time):
    h = int(time // 3600)
    m = int((time - h * 3600) // 60)
    s = int(time - h * 3600 - m * 60)
    if h > 0:
        line = '{}h{:02d}m'.format(h, m)
    else:
        line = '{}m{:02d}s'.format(m, s)

    return line

