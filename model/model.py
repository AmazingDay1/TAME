#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : zy.xiao
# @File     : model.py
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from model.mamba import MambaBlock
from mamba_ssm.ops.triton.layernorm import RMSNorm
from timm.models.layers import trunc_normal_, lecun_normal_
from model.attention import ConvergeAttention

import torch.nn.functional as F


class TFMamba(nn.Module):
    def __init__(self, num_cls, mode="train", t_f_tocken=True, abs_pos_embed=True, depth=12, cnn_region_feature=True):
        super(TFMamba, self).__init__()

        self.num_class = num_cls
        self.embed_dim = 192
        self.t_f_tocken = t_f_tocken
        self.abs_pos_embed = abs_pos_embed
        self.extend_tocken_num = 1

        # region_fusion
        self.region_feature = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.normal = nn.BatchNorm2d(32)
        self.silu = nn.SiLU()
        # time split patch
        if cnn_region_feature == True:
            self.t_patch_embed = PatchEmbed((224, 16), (14, 16), (14, 1), 4, self.embed_dim)
        else:
            self.t_patch_embed = PatchEmbed((64, 64), (4, 64), (4, 1), 4, self.embed_dim)
        t_num_patch = self.t_patch_embed.num_patch
        # frequency split patch
        if cnn_region_feature == True:
            self.f_patch_embed = PatchEmbed((224, 16), (224, 1), (1, 1), 4, self.embed_dim)
        else:
            self.f_patch_embed = PatchEmbed((64, 64), (64, 4), (1, 4), 4, self.embed_dim)
        f_num_patch = self.t_patch_embed.num_patch

        if self.t_f_tocken:
            self.t_tocken = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.f_tocken = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if self.abs_pos_embed:
            self.t_abs_pos_embed = nn.Parameter(torch.zeros(1, t_num_patch + self.extend_tocken_num, self.embed_dim))
            self.t_pos_drop_path = DropPath(0.0)
            self.f_abs_pos_embed = nn.Parameter(torch.zeros(1, f_num_patch + self.extend_tocken_num, self.embed_dim))
            self.f_pos_drop_path = DropPath(0.0)

        dpr = [x.item() for x in torch.linspace(0, 0.1, depth)]
        dpr = [0.0] + dpr
        self.drop_path = DropPath(0.1)
        self.t_backbone = nn.ModuleList([
            MambaBlock(
                self.embed_dim,
                d_state=16,
                norm_epsilon=1e-5,
                fused_add_norm=True,
                layer_idx=i,
                drop_path=dpr[i],
                t_f_kind="time"
            )
            for i in range(depth)
        ])
        self.f_backbone = nn.ModuleList([
            MambaBlock(
                self.embed_dim,
                d_state=16,
                norm_epsilon=1e-5,
                fused_add_norm=True,
                layer_idx=i,
                drop_path=dpr[i],
                t_f_kind="frequency"

            ) for i in range(depth)
        ])

        self.out_norm = RMSNorm(self.embed_dim)

        self.cross_attention_converge_t = ConvergeAttention()
        self.cross_attention_converge_f = ConvergeAttention()
        self.cross_attention_converge_tf = ConvergeAttention()


        self.cls_head1 = nn.Linear(self.embed_dim, 512)
        self.cls_head2 = nn.Linear(512, self.num_class)

        self.pos_head1 = nn.Linear(self.embed_dim, 512)
        self.pos_head2 = nn.Linear(512, 3)

        if mode == "train":
            # init weight
            self.t_patch_embed.apply(self.init_weights)
            self.f_patch_embed.apply(self.init_weights)
            self.t_backbone.apply(self.init_weights)
            self.f_backbone.apply(self.init_weights)

            self.cross_attention_converge_t.apply(self.init_weights)
            self.cross_attention_converge_f.apply(self.init_weights)
            self.cross_attention_converge_tf.apply(self.init_weights)

            self.cls_head1.apply(self.init_weights)
            self.cls_head2.apply(self.init_weights)
            self.pos_head1.apply(self.init_weights)
            self.pos_head2.apply(self.init_weights)
            if abs_pos_embed:
                trunc_normal_(self.t_abs_pos_embed, std=.02)
                trunc_normal_(self.f_abs_pos_embed, std=.02)
            if t_f_tocken:
                trunc_normal_(self.t_tocken, std=.02)
                trunc_normal_(self.f_tocken, std=.02)
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            # NOTE conv was left to pytorch default in my original init
            lecun_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):  # input->[4 64 16]

        if x.dim() == 3:  # in order to calculate gflops and parm
            x = x.unsqueeze(dim=0)

        t = self.t_patch_embed(x)
        B, M, D = t.shape
        f = self.f_patch_embed(x)
        if self.t_f_tocken:
            t_tocken = self.t_tocken.expand(B, -1, -1)
            f_tocken = self.f_tocken.expand(B, -1, -1)

            t = torch.cat([t, t_tocken], dim=1)
            f = torch.cat([f, f_tocken], dim=1)

        if self.abs_pos_embed:
            t = t + self.t_abs_pos_embed
            f = f + self.f_abs_pos_embed
            f = self.f_pos_drop_path(f)

        residual_t = None
        residual_f = None
        for t_block in self.t_backbone:
            t, residual_t = t_block(t, residual_t)
        for f_block in self.f_backbone:
            f, residual_f = f_block(f, residual_f)

        t1 = self.cross_attention_converge_t(t, f)
        fusion_feature = self.cross_attention_converge_tf(t1, f)
        fusion_feature = t + fusion_feature

        fusion_feature = fusion_feature[:, -1, :]

        cls = F.relu(self.cls_head1(fusion_feature))
        cls = self.cls_head2(cls)

        pos = F.relu(self.pos_head1(fusion_feature))
        pos = self.pos_head2(pos)

        return cls, pos


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, stride, in_channel, embed_dim, normal=None, flatten=True):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.grid_size = ((self.img_size[0] - self.patch_size[0]) // self.stride[0] + 1,
                          (self.img_size[1] - self.patch_size[1]) // self.stride[1] + 1)
        self.num_patch = self.grid_size[0] * self.grid_size[1]

        self.split_patch = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=stride)
        self.flatten = flatten
        self.normal = nn.BatchNorm2d(embed_dim) if normal else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input size doesn't match model."
        x = self.split_patch(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.normal(x)
        return x  # [1 16 192]
