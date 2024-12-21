#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : zy.xiao
# @File     : dataset.py
import os.path

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from dataloader.data_process import *


class AudioDataset(Dataset):
    def __init__(self, annotation_path, gt_cls_path, gt_postion_path, audio_path, mode="train"):
        super(AudioDataset, self).__init__()
        if mode == "train":
            with open(annotation_path, "r") as f:
                self.annotation_lines = f.readlines()  # 727  0/100.npy
        elif mode == "test":
            with open(annotation_path, "r") as f:
                self.annotation_lines = f.readlines()
        self.gt_cls_path = gt_cls_path
        self.gt_postion_path = gt_postion_path
        self.audio_path = audio_path

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        file_name = self.annotation_lines[index][:-1]

        gt_cls_path = os.path.join(self.gt_cls_path, file_name)
        gt_position_path = os.path.join(self.gt_postion_path, file_name)

        # load audio data
        audio = concat_audio(self.audio_path, file_name)  # mean=0 std=1  [15500 4]
        # audio = np.transpose(audio, [1, 0])  #  [4 15500]
        spectrogram = audio_to_spectrogram(audio)  # [4, 64 16]
        spectrogram = spectrogram.float()
        # spectrogram = torch.tensor(spectrogram, dtype=float)

        # load gt cls dada
        gt_cls = np.load(gt_cls_path)[0]
        gt_cls = torch.tensor(gt_cls)

        # # load gt position data
        gt_position = np.array(np.load(gt_position_path))
        gt_position = torch.from_numpy(gt_position).float()

        return spectrogram, gt_cls, gt_position
