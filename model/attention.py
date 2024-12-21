#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : zy.xiao
# @File     : attention.py
import torch
import torch.nn as nn


class ConvergeAttention(nn.Module):
    def __init__(self, dim=192, head=6, proj_drop=0.1):
        super().__init__()
        self.heads = head
        head_dim = dim//head
        self.scale = head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)  # 96
        self.q = nn.Linear(dim, dim, bias=True)  # TODO：测试需不需要对qkv进行映射
        self.kv = nn.Linear(dim, 2*dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(proj_drop)

    def forward(self, t, f):
        residual_x = t
        B, N, C = t.shape

        f = self.norm(f)
        t = self.norm(t)

        q = self.q(t).reshape(B, N, self.heads, C//self.heads).permute(0, 2, 1, 3).contiguous()  # [90 17 96] -> 90 17 6 16 -> [90 6 17 16]
        # kv = self.kv(f).reshape(B, -1, 2, self.heads // 2, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()  # [90 17 96] -> [90 17 2 3 16] -> [2 90 3 17 16]
        # k, v = kv[0], kv[1]  # [90 3 17 16]

        k, v = self.kv(f).chunk(2, dim=-1)  # TODO:对比复制的效果
        k = k.reshape(B, N, self.heads, C//self.heads).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, N, self.heads, C//self.heads).permute(0, 2, 1, 3).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale  # # [90 6 17 17]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1).contiguous()  # [90 3 17 17]@[90 3 17 16] -> [90 3 17 16]
        x = self.proj(x)
        x = self.attn_drop(x)

        x = x + residual_x

        return x
