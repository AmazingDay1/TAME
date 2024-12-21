#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : zy.xiao
# @File     : train.py

import argparse
import os

import torch

from dataloader.dataset import AudioDataset
from torch.utils.data import DataLoader
from model.model import TFMamba
from torch import optim
from tqdm import tqdm
import numpy as np
from functools import partial
import math


def main():
    parser = argparse.ArgumentParser(description="Position estimation based on Audio used by Mamba")

    parser.add_argument("--audio_path", type=str, default="/media/xiao/HIKSEMI/datasets/MMAUD/Data-M/audio_npy/", help="load data from the path")
    parser.add_argument("--gt_cls_path", type=str, default="/media/xiao/HIKSEMI/datasets/MMAUD/Data-M/label/", help="the gt type of uav")
    parser.add_argument("--gt_position_path", type=str, default="/media/xiao/HIKSEMI/datasets/MMAUD/Data-M/gt/", help="the gt 3d position of uav")

    parser.add_argument("--save_path", type=str, default="output/", help="the path to save model")

    parser.add_argument("--train_split_path", type=str, default="/media/xiao/HIKSEMI/datasets/MMAUD/Data-M/annotation/annotation_train_all/trainval.txt",
                        help="the file of train, format: cls/file_name.npy, eg:0/111.npy")
    parser.add_argument("--val_split_path", type=str, default="/media/xiao/HIKSEMI/datasets/MMAUD/Data-M/annotation/annotation_test_all/trainval.txt",
                        help="the file of train, format: cls/file_name.npy, eg:0/111.npy")

    parser.add_argument("--batch_size", type=int, default=64, help="data size of per batch")
    parser.add_argument("--train_epoch", type=int, default=200, help="number of training epochs")
    parser.add_argument("--workers", type=int, default=2, help="number of workers for dataloader")
    parser.add_argument("--gpu", type=str, default="cuda:0", help="training on the gpu device")
    parser.add_argument("--resume", type=str, default="", help="training on the gpu device")
    args = parser.parse_args()

    # set dataset and dataloader
    train_dataset = AudioDataset(args.train_split_path, args.gt_cls_path, args.gt_position_path, args.audio_path)
    test_dataset = AudioDataset(args.val_split_path, args.gt_cls_path, args.gt_position_path, args.audio_path)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)

    # set model
    model = TFMamba(num_cls=5, mode="train")
    # model = TFMamba(num_cls=5, mode="test")
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(device)

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    lr_fun = lr_adjust_fun("decay", 0.0001, 0.0001 * 0.01, args.train_epoch)

    # set loss
    pos_loss = torch.nn.L1Loss()
    cls_loss = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    for epoch in range(args.train_epoch):
        set_optim_lr(optimizer, lr_fun, epoch)
        print("current lr:", optimizer.param_groups[0]["lr"])

        train_loss = train_mode(model, train_loader, optimizer, pos_loss, cls_loss, device)
        valid_loss = valid_mode(model, test_loader, pos_loss, cls_loss, device)
        print("Epoch {}/{}, Train Loss: {}, Val Loss: {}".format(str(epoch + 1), args.train_epoch, str(train_loss), str(valid_loss)))

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(args.save_path, "best_model_{}_{}.pth".format(str(epoch + 1), str(valid_loss))))

        torch.save(model.state_dict(), os.path.join(args.save_path, "epoch{}_val_loss_{}.pth".format(str(epoch + 1), str(valid_loss))))


def train_mode(model, train_loader, optimizer, pos_loss, cls_loss, device):
    model.train()
    train_loss = 0
    for data in tqdm(train_loader, total=len(train_loader), unit="batch"):
        spectrogram, cls_gt, pos_gt = [d.to(device) for d in data]
        optimizer.zero_grad()
        cls_pred, pos_pred = model(spectrogram)  # [90 60]  [90 3]

        loss_cls = cls_loss(cls_pred, cls_gt)
        loss_pos = pos_loss(pos_pred, pos_gt)
        print("loss cls:", loss_cls)
        print("loss pos:", loss_pos)
        total_loss = loss_cls + 2*loss_pos
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()
    return train_loss/len(train_loader)


def set_optim_lr(optim, lr_adjust_fun, epoch):
    lr = lr_adjust_fun(epoch)
    for param_group in optim.param_groups:
        param_group["lr"] = lr


def pos_loss(pos_pred, pos_label):
    # print(y_true.shape,y_pred.shape)
    mse_loss = torch.nn.L1Loss()
    mse_loss = mse_loss(pos_pred, pos_label)
    return mse_loss


def lr_adjust_fun(lr_decay_type, lr, min_lr, total_iters,
                  warmup_iters_ratio=0.05,
                  warmup_lr_ratio=0.1,
                  no_aug_iter_ratio=0.05,
                  step_num=10):

    def warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi*(iters - warmup_total_iters) / (total_iters - warmup_total_iters -no_aug_iter)))
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        fun = partial(warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        fun = partial(step_lr, lr, decay_rate, step_size)
    return fun


def valid_mode(model, valid_loader, pos_loss, cls_loss, device):
    model.eval()
    valid_loss = 0
    i = 0
    with torch.no_grad():
        for data in tqdm(valid_loader, total=len(valid_loader), unit="batch"):
            spectrogram, cls_gt, pos_gt = [d.to(device) for d in data]
            cls_pred, pos_pred = model(spectrogram)

            loss_cls = cls_loss(cls_pred, cls_gt)
            loss_pos = pos_loss(pos_pred, pos_gt)
            total_loss = loss_cls + loss_pos
            valid_loss += total_loss.item()
    return valid_loss/len(valid_loader)


if __name__ == "__main__":
    main()

