#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : zy.xiao
# @File     : train.py

import argparse
import math
import os

import torch

from dataloader.dataset import AudioDataset
from torch.utils.data import DataLoader
from model.model import TFMamba
from torch import optim
from tqdm import tqdm
import numpy as np
from torchsummary import  summary
from thop import clever_format, profile
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Position estimation based on Audio used by Mamba")
    parser.add_argument("--audio_path", type=str, default="/media/xiao/HIKSEMI/datasets/MMAUD/Data-M/audio_npy/", help="load data from the path")
    parser.add_argument("--gt_cls_path", type=str, default="/media/xiao/HIKSEMI/datasets/MMAUD/Data-M/label/", help="the gt type of uav")
    parser.add_argument("--gt_position_path", type=str, default="/media/xiao/HIKSEMI/datasets/MMAUD/Data-M/gt", help="the gt 3d position of uav")

    parser.add_argument("--val_split_path", type=str, default="/media/xiao/HIKSEMI/datasets/MMAUD/Data-M/annotation/annotation_test_all/trainval.txt",
                        help="the file of train, format: cls/file_name.npy, eg:0/111.npy")
    parser.add_argument("--save_path", type=str, default="output/", help="the path to save model")

    parser.add_argument("--batch_size", type=int, default=64, help="data size of per batch")
    parser.add_argument("--workers", type=int, default=1, help="number of workers for dataloader")
    parser.add_argument("--gpu", type=str, default="cuda:0", help="training on the gpu device")
    parser.add_argument("--resume", type=str, default="output/best_model_177_0.34104672290904575.pth", help="training on the gpu device")   #output/best_model1.683644413948059.pth
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # set dataset and dataloader
    test_dataset = AudioDataset(args.val_split_path, args.gt_cls_path, args.gt_position_path, args.audio_path)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    # set model
    model = TFMamba(num_cls=5, mode="test")
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    valid_mode(model, test_loader, device)


def pos_loss(pos_pred, pos_label):
    # print(y_true.shape,y_pred.shape)
    mse_loss = torch.nn.L1Loss()
    mse_loss = mse_loss(pos_pred, pos_label)
    return mse_loss

def valid_mode(model, valid_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_pred_pos = []
    all_gt_pos = []
    # with torch.no_grad():
    for data in tqdm(valid_loader, total=len(valid_loader), unit="batch"):
        spectrogram, cls_gt, pos_gt = [d.to(device) for d in data]

        cls_pred, pos_pred = model(spectrogram)

        _, preds = torch.max(cls_pred, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(cls_gt.cpu().numpy())

        p_and_gt = np.array(pos_pred.cpu().detach().numpy())
        all_pred_pos.append(p_and_gt)
        all_gt_pos.append(pos_gt.cpu().detach().numpy())
    result_pos = np.concatenate(all_pred_pos, axis=0)
    gt_p = np.concatenate(all_gt_pos, axis=0)

    accuracy(all_preds, all_labels)  # calculate the accuracy and confusion_matrix
    average_pos_err(result_pos, gt_p)  # calculate the APE
    pos_x_y_z_err(result_pos, gt_p)   # calculate the Dx Dy Dz
    draw_pos(result_pos, gt_p, all_labels, cls_num=0, cls_name="Mavic2")
    draw_pos(result_pos, gt_p, all_labels, cls_num=1, cls_name="Mavic3")
    draw_pos(result_pos, gt_p, all_labels, cls_num=2, cls_name="Phantom4")
    draw_pos(result_pos, gt_p, all_labels, cls_num=3, cls_name="Avata")
    draw_pos(result_pos, gt_p, all_labels, cls_num=4, cls_name="M300")

    gflops_params(model)
    # caculate_flop_param(model)


def save_csv_pred(result_pos, cls_name, ground_truth_timestamps):

    # ground_truth_timestamps = [i for i in range(0, num, 1)]
    original_poses_df = pd.DataFrame(result_pos, columns=['x', 'y', 'z'], index=ground_truth_timestamps)
    original_poses_df.index.name = 'timestamp'
    original_poses_csv_path = os.path.join('result_csv/', '{}_pred.csv'.format(cls_name))
    original_poses_df.to_csv(original_poses_csv_path)


def save_csv_gt(result_pos, cls_name, ground_truth_timestamps):

    # ground_truth_timestamps = [i for i in range(0, num, 1)]
    original_poses_df = pd.DataFrame(result_pos, columns=['x', 'y', 'z'], index=ground_truth_timestamps)
    original_poses_df.index.name = 'timestamp'
    original_poses_csv_path = os.path.join('result_csv/', '{}_gt.csv'.format(cls_name))
    original_poses_df.to_csv(original_poses_csv_path)

def draw_two_line(points_pred, points_gt, cls_name, axis_limits, ground_truth_timestamps):
    save_csv_pred(points_pred, cls_name, ground_truth_timestamps)
    save_csv_gt(points_gt, cls_name, ground_truth_timestamps)

def draw_pos(pos_pred, pos_gt, all_labels, cls_num, cls_name):
    # to save the range of x y z
    gt_max = np.array([-np.inf, -np.inf, -np.inf])
    gt_min = np.array([np.inf, np.inf, np.inf])

    for idx in range(len(all_labels)):
        if all_labels[idx] == cls_num:
            gt_max = np.maximum(gt_max, pos_gt[idx])  # calculate the range of x y z
            gt_min = np.minimum(gt_min, pos_gt[idx])  #

    # calculate the center of trajectory
    center_point = (gt_max + gt_min) / 2
    center_point = center_point[np.newaxis,:]
    max_range = np.max(gt_max - center_point)
    # calculate the range of every axis
    axis_limits = np.array([center_point - max_range, center_point + max_range])  # [2, 3]
    axis_limits = axis_limits.squeeze(axis=1).transpose(1, 0)

    start_idx = 20000
    end_idx = -1
    for idx in range(len(all_labels)):
        if all_labels[idx] == cls_num:
            if start_idx > idx:
                start_idx = idx
            if end_idx < idx:
                end_idx = idx
        else:
            continue
    gt_points = pos_gt[start_idx:end_idx, :]  # select the class rusult to draw
    pred_points = pos_pred[start_idx:end_idx, :]
    with open("result_csv/test.txt", "r") as f:
        ground_truth_timestamps = f.readlines()  # read_timestamps
    ground_truth_timestamps = [float(i.strip()) for i in ground_truth_timestamps]
    ground_truth_timestamps = ground_truth_timestamps[start_idx:end_idx]
    draw_two_line(pred_points, gt_points, cls_name, axis_limits, ground_truth_timestamps)

def accuracy(all_preds, all_labels):
    # # Example predictions and targets
    # predictions = torch.tensor([0, 1, 2, 1, 0, 2])
    # targets = torch.tensor([0, 1, 1, 1, 0, 2])
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    correct = (all_preds == all_labels).astype(int).sum().item()
    acc = correct / len(all_labels)
    print("Accuracy: %s" % acc)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    cm_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    print("Confusion Matrix")
    print(cm_normalized)
    return cm_normalized


def average_pos_err(pos_pred, pos_gt):
    N, _ = pos_pred.shape
    distance = (pos_pred - pos_gt)**2  # L2 Normal
    # APE = np.sum(math.sqrt(pos_pred**2)) / N
    APE = np.sqrt(distance[:,0] + distance[:,1] + distance[:,2]).sum() / N
    print("APE: %s" % APE)

def pos_x_y_z_err(pos_pred, pos_gt):
    N, _ = pos_pred.shape
    x_err = (np.abs(pos_pred[:, 0] - pos_gt[:, 0])).sum() / N
    y_err = (np.abs(pos_pred[:, 1] - pos_gt[:, 1])).sum() / N
    z_err = (np.abs(pos_pred[:, 2] - pos_gt[:, 2])).sum() / N
    print("Dx:", x_err)
    print("Dy:", y_err)
    print("Dz:", z_err)

def gflops_params(model):
    input_shape = [224, 16]
    num_class = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary(model, (4, input_shape[0], input_shape[1]))

    input = torch.randn(1, 4, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(model.to(device), (input), verbose=False)
    flops = 2*flops
    flops, params = clever_format([flops, params], "%.3f")
    print("Total GFLOPs: %s" % (flops))
    print("Total params: %s" % (params))




if __name__ == "__main__":
    main()

