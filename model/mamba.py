#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : zy.xiao
# @File     : mamba.py
import math

import torch
import torch.nn as nn
import triton
import triton.language as tl
from einops import rearrange, repeat
# import selective_scan_cuda_oflex
# import selective_scan_cuda
# import selective_scan_cuda_core
from functools import partial

try:
    import selective_scan_cuda_oflex
except ImportError:
    WITH_SELECTIVESCAN_OFLEX = False
    print("Can not import selective_scan_cuda_oflex. This affects speed.", flush=True)
try:
    import selective_scan_cuda_core
except ImportError:
    WITH_SELECTIVESCAN_CORE = False
try:
    import selective_scan_cuda
except ImportError:
    WITH_SELECTIVESCAN_MAMBA = False

from mamba_ssm.ops.triton.layernorm import RMSNorm, rms_norm_fn
from timm.models.layers import DropPath

class MambaBlock(nn.Module):
    def __init__(self,
                 d_model,
                 # ssm_cfg=None,
                 d_state=16,
                 norm_epsilon=1e-5,
                 d_conv=4,
                 expand=2,
                 dt_rank="auto",
                 # dt_min=0.001,
                 # dt_max=0.1,
                 # dt_init="random",
                 # dt_scale=1.0,
                 # dt_init_floor=1e-4,
                 # conv_bias=True,
                 # bias=False,
                 fused_add_norm=True,
                 layer_idx=None,
                 drop_path=None,
                 t_f_kind=""

                 ):
        super(MambaBlock, self).__init__()
        self.d_model = d_model  # 96
        self.d_state = d_state
        self.d_conv = d_conv  # 4
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model/16) if dt_rank == "auto" else dt_rank

        self.norm_cls = partial(RMSNorm, eps=norm_epsilon)
        self.norm = self.norm_cls(self.d_model)

        self.fused_add_norm_fn = rms_norm_fn
        self.layer_idx = layer_idx

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.in_proj = nn.Linear(self.d_model, self.d_inner*2, bias=False)  # 需要拆分成x z 所以self.d_inner*2

        self.activate = nn.SiLU()
        self.t_f_kind = t_f_kind
        if t_f_kind == "time":
            self.con2d = nn.Conv2d(self.d_model * 2, self.d_model * 2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=self.d_model * 2)
            self.ssm = SSM(self.d_model, self.d_state, self.d_inner, self.dt_rank, t_f_kind="time")

        elif t_f_kind == "frequency":
            self.con2d = nn.Conv2d(self.d_model * 2, self.d_model * 2, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), groups=self.d_model * 2)
            self.ssm = SSM(self.d_model, self.d_state, self.d_inner, self.dt_rank, t_f_kind="frequency")

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x, residual):
        if residual is None:
            x, residual = self.fused_add_norm_fn(x,
                                                 self.norm.weight,
                                                 self.norm.bias,
                                                 residual=residual,
                                                 prenorm=True,
                                                 residual_in_fp32=True,
                                                 eps=self.norm.eps)
        else:
            x, residual = self.fused_add_norm_fn(self.drop_path(x),
                                                 self.norm.weight,
                                                 self.norm.bias,
                                                 residual=residual,
                                                 prenorm=True,
                                                 residual_in_fp32=True,
                                                 eps=self.norm.eps)

        x = self.in_proj(x)  # [90 17 96]->[90 17 384]
        x, z = x.chunk(2, dim=-1)  # [90 17 192]

        if self.t_f_kind == "time":
            x = x.unsqueeze(2).permute(0, 3, 1, 2).contiguous()
            x = self.con2d(x)
            x = x.squeeze(3)

        elif self.t_f_kind == "frequency":
            x = x.unsqueeze(1).permute(0, 3, 1, 2).contiguous()
            x = self.con2d(x)
            x = x.squeeze(2)

        x = self.activate(x)
        y = self.ssm.forward_fn(x)

        y = y * self.activate(z)

        out = self.out_proj(y)

        return out, residual


class SSM(nn.Module):
    def __init__(self,
                 d_mode,
                 d_state,
                 d_inner,
                 rank,
                 dt_scale=1.0,
                 dt_max=0.1,
                 dt_min=0.001,
                 dt_init_floor=0.0001,
                 t_f_kind="",
                 device=None):
        super(SSM, self).__init__()
        self.dt_max = dt_max

        self.d_model = d_mode  # 96
        self.d_state = d_state  # 16
        self.d_inner = d_inner  # 192
        self.dt_rank = rank  # 6   96/6
        self.t_f_kind = t_f_kind

        self.out_norm = nn.LayerNorm(self.d_model * 2)

        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                   "n -> d n",
                   d=self.d_inner).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # (192)
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj_weight = nn.Parameter(self.dt_proj.weight)  # (rank=12 192)
        self.dt_proj_bias = nn.Parameter(self.dt_proj.bias)  # (192)
        del self.dt_proj

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False)  # 192 28

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

    def forward_fn(self, xs):
        B, D, L = xs.shape  # [90 192 1 16]
        D, N = self.A_log.shape  # 192 16
        D, R = self.dt_proj_weight.shape  # 192 6

        x_dbl = self.x_proj(rearrange(xs, "b d l -> b l d"))  # [90 192 17] [38 192] -> [90  17 38]
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)  # [90 17 6][90 17 16][90 17 16]
        dts = torch.einsum("b l r, d r -> b d l", dts, self.dt_proj_weight)  # [90 17 6] [192 6] ->[90 192 17]
        xs = xs.view(B, -1, L).to(torch.float32)  # 90 192 17
        dts = dts.contiguous().view(B, -1, L).to(torch.float32)  # 1 192 17tocken
        As = -self.A_log.to(torch.float).exp()   # 192 16state
        Ds = self.D.to(torch.float)  # (192)
        Bs = Bs.contiguous().view(B, N, L).to(torch.float32)  # 1 16state 16tocken
        Cs = Cs.contiguous().view(B, N, L).to(torch.float32)  # 1 16state 16tocken
        delta_bias = self.dt_proj_bias.view(-1).to(torch.float)  # 192

        # y = selective_scan_fn(xs, dts, As, Bs, Cs, Ds, delta_bias=delta_bias, delta_softplus=True, ssoflex=True, backend="oflex")
        Bs = Bs.unsqueeze(1)
        Cs = Cs.unsqueeze(1)
        y = selective_scan_fn(xs, dts, As, Bs, Cs, Ds, delta_bias=delta_bias, delta_softplus=True, ssoflex=True, backend="mamba")

        y = y.permute(0, 2, 1).contiguous()  # [90 17 1 192]
        y = self.out_norm(y)

        return y.to(xs.dtype)


def selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=None):
    with_cuda = True
    fn = selective_scan_torch if backend == "torch" else SelectiveScanCuda.apply # implement by torch
    # fn = selective_scan_torch # implement by triton
    return fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend)


class SelectiveScanCuda(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True, backend=None):
        ctx.delta_softplus = delta_softplus
        # backend = "oflex" if WITH_SELECTIVESCAN_OFLEX and (backend is None) else backend
        # backend = "core" if WITH_SELECTIVESCAN_CORE and (backend is None) else backend
        # backend = "mamba" if WITH_SELECTIVESCAN_MAMBA and (backend is None) else backend
        ctx.backend = backend
        if backend == "oflex":
            out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        elif backend == "core":
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        elif backend == "mamba":

            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        backend = ctx.backend
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if backend == "oflex":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        elif backend == "core":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        elif backend == "mamba":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False
            )
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None


def selective_scan_torch(u,  # [B C L]
                         delta,  # [B C L] [1 192 1*16]
                         A,  # [192 16]
                         B,  # [90 1 16 1*17]
                         C,  # [90 1 16 1*17]
                         D,  # [192]
                         delta_bias,  # [192]
                         delta_softplus,
                         ssoflex=None,
                         backend=None):
    dtyper_in = u.dtype
    _, inner_dim, _ = u.shape
    Bs, G, N, L = B.shape
    dim = u.shape[1]  # 192

    assert u.shape == (Bs, inner_dim, L)  # [1 192 1*17]
    assert delta.shape == (Bs, inner_dim, L)
    assert A.shape == (inner_dim, N)  # 每张图像都是相同的
    assert C.shape == B.shape  # [90 1 16 1*16]

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)

    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    B = B.view(Bs, 1, N, L).repeat(1, dim, 1, 1).view(Bs, dim, N, L)  # [1 192 16 1*17]
    C = C.view(Bs, 1, N, L).repeat(1, dim, 1, 1).view(Bs, dim, N, L)  # [1 192 16 1*17]
    deltaA = torch.exp(torch.einsum("bdl, dn->bdln", delta, A))
    deltaB = torch.exp(torch.einsum("bdl, bdnl->bdln", delta, B))
    deltaBu = torch.einsum("bdln, bdl ->bdln" , deltaB, u)

    hidden_state = A.new_zeros((Bs, dim, N))
    ys = []
    for i in range(L):
        hidden_state = deltaA[:, :, i, :] * hidden_state  + deltaBu[:, :, i, :]  # [90 192 16]
        y = torch.einsum("bdn, bdn->bd", hidden_state, C[:, :, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2)

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=dtyper_in)


def scan_fn(x, in_channel=True, out_channel=True, one_by_one=False, with_triton=True):
    Scan = CrossScanTriton if with_triton else ScanTorch # implement by torch
    return Scan.apply(x, in_channel, out_channel, one_by_one)    # x [1 192 1 16]

class ScanTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, in_channel, out_channel, one_by_one):
        ctx.in_channel = in_channel
        ctx.out_channel = out_channel
        ctx.one_by_one = one_by_one

        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        y = x.flatten(2, 3)  # [1 192 16]
        # fn = cross_scan1b1_fwd  # 如果是1d 的con
        # y = fn(x, in_channel, out_channel)
        return y  # [1 192 16]

    def backward(ctx, ys):
        # in_channel = ctx.in_channel
        # out_channel = ctx.out_channel
        # one_by_one = ctx.one_by_one
        B, C, H, W = ctx.shape

        # B, D, H, W = ys.shape
        # ys = ys.view(B, -1, C, H, W)
        # ys = ys.view(B, D, -1)
        y = ys.view(B, -1, H, W)
        return y, None, None, None, None

class CrossScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, in_channel, out_channel, one_by_one):
        if one_by_one:
            if in_channel:
                B, _, C, H, W = x.shape
            else:
                B, H, W, _, C = x.shape
        else:
            if in_channel:
                B, C, H, W = x.shape  # [1 192 17 1]
            else:
                B, H, W, C = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = 1, 32, 32
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)

        ctx.in_channel = in_channel
        ctx.out_channel = out_channel
        ctx.one_by_one = one_by_one
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)

        y = x.new_empty((B, C, H*W)) if out_channel else x.new_empty((B, H*W, C))  # [1 192 1*16]
        triton_scan_flex[(NH*NW, NC, B)](x.contiguous(),
                                       y,
                                       (0 if in_channel else 1),
                                       (0 if out_channel else 1),
                                       0,
                                       (0 if one_by_one else 1),
                                        0,
                                       BC,
                                       BH,
                                       BW,
                                       C,
                                       H,
                                       W,
                                       NH,
                                       NW
                                       )
        return y

@triton.jit
def triton_scan_flex(
    x, # (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
    y, # (B, 4, C, H, W) | (B, H, W, 4, C)
    x_layout: tl.constexpr,
    y_layout: tl.constexpr,
    operation: tl.constexpr,
    onebyone: tl.constexpr,
    scans: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    # x_layout = 0
    # y_layout = 1 # 0 BCHW, 1 BHWC
    # operation = 0 # 0 scan, 1 merge
    # onebyone = 0 # 0 false, 1 true
    # scans = 0 # 0 cross scan, 1 unidirectional, 2 bidirectional

    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    HWRoute0 = i_h * BH * DW  + tl.arange(0, BH)[:, None] * DW + i_w * BW + tl.arange(0, BW)[None, :]
    # HWRoute1 = i_w * BW * DH + tl.arange(0, BW)[None, :] * DH + i_h * BH + tl.arange(0, BH)[:, None]  # trans
    # HWRoute2 = (NH - i_h - 1) * BH * DW  + (BH - 1 - tl.arange(0, BH)[:, None]) * DW + (NW - i_w - 1) * BW + (BW - 1 - tl.arange(0, BW)[None, :]) + (DH - NH * BH) * DW + (DW - NW * BW) # flip
    # HWRoute3 = (NW - i_w - 1) * BW * DH  + (BW - 1 - tl.arange(0, BW)[None, :]) * DH + (NH - i_h - 1) * BH + (BH - 1 - tl.arange(0, BH)[:, None]) + (DH - NH * BH) + (DW - NW * BW) * DH  # trans + flip

    if scans == 1:
        HWRoute1 = HWRoute0
        # HWRoute2 = HWRoute0
        # HWRoute3 = HWRoute0
    elif scans == 2:
        HWRoute1 = HWRoute0
        HWRoute3 = HWRoute2

    _tmp1 = DC * DH * DW

    y_ptr_base = y + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if y_layout == 0 else i_c * BC)
    if y_layout == 0:
        p_y1 = y_ptr_base + HWRoute0
        # p_y2 = y_ptr_base + _tmp1 + HWRoute1
        # p_y3 = y_ptr_base + 2 * _tmp1 + HWRoute2
        # p_y4 = y_ptr_base + 3 * _tmp1 + HWRoute3
    else:
        p_y1 = y_ptr_base + HWRoute0 * 4 * DC
        # p_y2 = y_ptr_base + DC + HWRoute1 * 4 * DC
        # p_y3 = y_ptr_base + 2 * DC + HWRoute2 * 4 * DC
        # p_y4 = y_ptr_base + 3 * DC + HWRoute3 * 4 * DC

    if onebyone == 0:
        x_ptr_base = x + i_b * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
        if x_layout == 0:
            p_x = x_ptr_base + HWRoute0
        else:
            p_x = x_ptr_base + HWRoute0 * DC

        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _x = tl.load(p_x + _idx_x, mask=_mask_hw)
                tl.store(p_y1 + _idx_y, _x, mask=_mask_hw)
                # tl.store(p_y2 + _idx_y, _x, mask=_mask_hw)
                # tl.store(p_y3 + _idx_y, _x, mask=_mask_hw)
                # tl.store(p_y4 + _idx_y, _x, mask=_mask_hw)
        elif operation == 1:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _y1 = tl.load(p_y1 + _idx_y, mask=_mask_hw)
                # _y2 = tl.load(p_y2 + _idx_y, mask=_mask_hw)
                # _y3 = tl.load(p_y3 + _idx_y, mask=_mask_hw)
                # _y4 = tl.load(p_y4 + _idx_y, mask=_mask_hw)
                # tl.store(p_x + _idx_x, _y1 + _y2 + _y3 + _y4, mask=_mask_hw)
                tl.store(p_x + _idx_x, _y1, mask=_mask_hw)

    else:
        x_ptr_base = x + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
        if x_layout == 0:
            p_x1 = x_ptr_base + HWRoute0
            # p_x2 = p_x1 + _tmp1
            # p_x3 = p_x2 + _tmp1
            # p_x4 = p_x3 + _tmp1
        else:
            p_x1 = x_ptr_base + HWRoute0 * 4 * DC
            # p_x2 = p_x1 + DC
            # p_x3 = p_x2 + DC
            # p_x4 = p_x3 + DC

        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                tl.store(p_y1 + _idx_y, tl.load(p_x1 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                # tl.store(p_y2 + _idx_y, tl.load(p_x2 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                # tl.store(p_y3 + _idx_y, tl.load(p_x3 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                # tl.store(p_y4 + _idx_y, tl.load(p_x4 + _idx_x, mask=_mask_hw), mask=_mask_hw)
        else:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                tl.store(p_x1 + _idx_x, tl.load(p_y1 + _idx_y), mask=_mask_hw)
                # tl.store(p_x2 + _idx_x, tl.load(p_y2 + _idx_y), mask=_mask_hw)
                # tl.store(p_x3 + _idx_x, tl.load(p_y3 + _idx_y), mask=_mask_hw)
                # tl.store(p_x4 + _idx_x, tl.load(p_y4 + _idx_y), mask=_mask_hw)
# def scan_fwd(x, in_channel, out_channel):
#     B, C, H, W = x.shape  # [1 192 1 16]
#     y = x.flatten(2, 3)  # [1 192 16]
#
#     return y




