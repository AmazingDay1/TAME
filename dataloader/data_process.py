#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : zy.xiao
# @File     : data_process.py
import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio.transforms as audio_trans
from torchvision import transforms


def concat_audio(audio_path, file_name):
    parts = file_name.split("/")  # 0/11.npy
    index = parts[-1][:-4]  # 11
    audio_data = np.load(os.path.join(audio_path, file_name))  # current time data [3100 4]
    return audio_data


def audio_to_spectrogram(audio, sr=48000, spectrogram_process_mode=1, min_frequency=10, max_frequency=3000):
    mel_spectrogram = audio_trans.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=1024,
        n_mels=200,
        pad_mode="constant",
        norm="slaney",
        mel_scale="slaney",
        power=2,
        # f_min=min_frequency
    )

    audio_data = torch.tensor(audio, dtype=torch.float32)
    spectrogram = mel_spectrogram(audio_data)  # [4 128 16]  [4 200 10]

    # # vis mel img
    # vis_audio = audio_data[0].unsqueeze(0)
    # mel_spec = spectrogram[0].detach().numpy()
    # plt.figure(figsize=(10, 4))
    # plt.axis('off')
    # plt.imshow(mel_spec, aspect="auto", origin="lower", cmap="viridis")
    # plt.imshow(mel_spec, aspect="auto", origin="lower",
    #            extent=[0, vis_audio.size(1)/sr, 0, sr / 2], cmap="viridis")
    # plt.colorbar(format="%+2.0f dB")
    # plt.title("Mel Spectrogram")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Frequency (Hz)")
    # plt.tight_layout()
    # # save img
    # plt.savefig("visaulization/mel_spectrogram_wo.svg")
    # plt.close()

    if spectrogram_process_mode == 1:
        spectrogram = scale_processing(spectrogram)  # scale 0~1
    elif spectrogram_process_mode == 0:
        spectrogram = normalization_processing(spectrogram)  # (data-m)/s
    transform = transforms.Resize((224, 16), antialias=True)

    spectrogram = transform(spectrogram)
    return spectrogram  # [4 224 16]


def scale_processing(data):
    """
    scale the data to [0, 1]
    :param data:
    :return:
    """
    for i in range(data.shape[0]):
        data_min = torch.min(data[i, :])
        data_max = torch.max(data[i, :])
        data[i, :] = (data[i, :] - data_min) / (data_max - data_min)
    return data


def normalization_processing(data):
    data_m = torch.mean(data)
    data_s = torch.std(data)
    data = (data - data_m) / data_s
    return data