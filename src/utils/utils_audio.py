#!/usr/bin/env python
# coding=utf-8
#@summerzhao: 2021/5/14
import numpy as np
import librosa

from src.config import cfg

def get_audio_magphase(y, n_fft, hop_length, win_length, window, normalize):
    # (*, 276)
    # Short-time Fourier transform (STFT)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)
    #spect = torch.FloatTensor(spect)

    if normalize:
        mean = spect.mean()
        std = spect.std()
        spect = spect-mean
        spect = np.divide(spect, std)
    spect = spect.transpose(1, 0)
    #print('get audio with magphase feature: ', spect.shape)
    return spect

def get_audio_fbank(y, sr, n_mels, n_fft, hop_length):
    # (*, 40)
    feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels = n_mels, n_fft=n_fft, hop_length=hop_length, power=1.)
    feat = np.log(feat + 1e-6)

    feat = [feat]

    feat = np.concatenate(feat, axis=0)
    if cfg.audio_feature == 'cmvn':
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]
    # 到底要不要翻转？
    #feat = np.swapaxes(feat, 0, 1)
    feat = feat.astype('float32')
    #print('get audio with fbank feature: ', feat.shape)
    return feat

def get_audio_mfcc(y, sr, n_mfcc, n_fft, hop_length):
    # (*, 13)
    mfcc_f = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc_f, width = 3)  # 一阶差分
    mfcc_delta2 = librosa.feature.delta(mfcc_f, width = 3, order = 2)  # 二阶差分
    mfccs = np.vstack((mfcc_f, mfcc_delta, mfcc_delta2))  # 整合成39维MFCC特征
    mfccs = np.swapaxes(mfccs, 0, 1).astype('float32')
    #print('get audio with mfcc feature: ', mfccs.shape)
    return mfccs

def get_audio_all(y, sr, n_fft, hop_length, window, normalize = True):
    win_length = n_fft
    magphase = get_audio_magphase(y, n_fft, hop_length, win_length, window, normalize)
    fbank = get_audio_fbank(y, sr, n_mels = 40, n_fft = n_fft, hop_length = hop_length)
    mfcc = get_audio_mfcc(y, sr, n_mfcc = 13, n_fft = n_fft, hop_length = hop_length)
    features = np.concatenate((magphase, fbank, mfcc), axis = -1)
    #print(magphase.shape, fbank.shape, mfcc.shape, features.shape)
    return features


