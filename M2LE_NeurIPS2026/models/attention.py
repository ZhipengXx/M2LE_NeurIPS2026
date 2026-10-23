import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.ndimage
import numpy as np
from typing import Tuple

def get_attn_fn(attn_fn_name):
    name = attn_fn_name.lower()
    options = {
               'glora': Attention_GLoRA,
               'hlora': Attention_HLoRA,
               'elora': Attention_LoRA,
               }
    return options[name]


class Attention_LoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.rank = r


        self.lora_A_k = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_k = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])

        self.lora_M_k = nn.ParameterList([nn.Parameter(torch.zeros(dim, dim), requires_grad=False) for _ in range(n_tasks)])
        self.lora_M_v = nn.ParameterList([nn.Parameter(torch.zeros(dim, dim), requires_grad=False) for _ in range(n_tasks)])
        self.ajto_k = _AjtoStats(dim)
        self.ajto_v = _AjtoStats(dim)

        self.register_buffer("H_sum_k", torch.zeros(dim, dim))
        self.register_buffer("G_sum_k", torch.zeros(dim, dim))
        self.register_buffer("H_sum_v", torch.zeros(dim, dim))
        self.register_buffer("G_sum_v", torch.zeros(dim, dim))
        self.register_buffer("pre_k", torch.zeros(dim, dim))
        self.register_buffer("pre_v", torch.zeros(dim, dim))


        self.collect_fisher = False
        self._fisher_delta_k = None
        self._fisher_delta_v = None
        self.use_pre_weight = False


        self.rank = r

        self.matrix = torch.zeros(dim ,dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(dim ,dim)
        self.n_cur_matrix = 0

    def init_param(self):

        with torch.no_grad():

            for t in range(len(self.lora_B_k)):
                nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_k[t].weight)
                nn.init.zeros_(self.lora_B_v[t].weight)


    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map


    def forward(self, x, task, register_hook=False, get_feat=False,get_cur_feat=False):
        # x size [128,197,768]
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_matrix + x.shape[0]*x.shape[1])
            self.n_matrix += x.shape[0]*x.shape[1]
        if get_cur_feat:
            # average self-attention
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0]*x.shape[1])
            self.n_cur_matrix += x.shape[0]*x.shape[1]

        B, N, C = x.shape
        # qkv: [B, N, 3*C] -> [3, B, heads, N, head_dim], where C = heads * head_dim.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)


        delta_k = self.lora_B_k[task].weight @ self.lora_A_k[task].weight
        delta_v = self.lora_B_v[task].weight @ self.lora_A_v[task].weight

        if delta_k.dtype != x.dtype:
            delta_k = delta_k.to(dtype=x.dtype)
            delta_v = delta_v.to(dtype=x.dtype)


        if self.collect_fisher:
            delta_k.retain_grad()
            delta_v.retain_grad()
            self._fisher_delta_k = delta_k
            self._fisher_delta_v = delta_v
        else:
            self._fisher_delta_k = None
            self._fisher_delta_v = None

        pre_k = self.pre_k
        pre_v = self.pre_v
        if pre_k.device != x.device or pre_k.dtype != x.dtype:
            pre_k = pre_k.to(device=x.device, dtype=x.dtype)
            pre_v = pre_v.to(device=x.device, dtype=x.dtype)

        use_theta = (self.ajto_k.use_theta and self.ajto_v.use_theta and not self.collect_fisher)
        if use_theta and self.use_pre_weight:
            cur_k = torch.zeros_like(pre_k)
            cur_v = torch.zeros_like(pre_v)
        elif use_theta:
            cur_k = self.lora_M_k[task].to(device=x.device, dtype=x.dtype)
            cur_v = self.lora_M_v[task].to(device=x.device, dtype=x.dtype)
        else:
            cur_k = delta_k
            cur_v = delta_v

        weight_k = pre_k + cur_k
        weight_v = pre_v + cur_v


        add_k = F.linear(x, weight_k)
        add_v = F.linear(x, weight_v)


        k = k + add_k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + add_v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_matrix(self, task):
        matrix_k = torch.mm(self.lora_B_k[task].weight, self.lora_A_k[task].weight)
        matrix_v = torch.mm(self.lora_B_v[task].weight, self.lora_A_v[task].weight)


        return matrix_k, matrix_v


    def get_pre_matrix(self, task):
        with torch.no_grad():

            weight_k = torch.stack([(self.lora_U_B_k[t].weight @ torch.diag(self.lora_S_B_k[t]) @ self.lora_Vh_B_k[t].weight @ self.lora_U_A_k[t].weight @ torch.diag(self.lora_S_A_k[t]) @ self.lora_Vh_A_k[t].weight) for t in range(task)], dim=0).sum(dim=0)
            weight_v = torch.stack([(self.lora_U_B_v[t].weight @ torch.diag(self.lora_S_B_v[t]) @ self.lora_Vh_B_v[t].weight @ self.lora_U_A_v[t].weight @ torch.diag(self.lora_S_A_v[t]) @ self.lora_Vh_A_v[t].weight) for t in range(task)], dim=0).sum(dim=0)


        return weight_k, weight_v


    def ajto_begin(self, ridge_lambda=1e-5):
        with torch.no_grad():
            self.ajto_k.begin(ridge_lambda)
            self.ajto_v.begin(ridge_lambda)


    def ajto_end(self):
        with torch.no_grad():
            self.ajto_k.end()
            self.ajto_v.end()
            self.use_pre_weight = False


    @property
    def ajto_using_theta(self):
        return self.ajto_k.use_theta and self.ajto_v.use_theta


class Attention_GLoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.rank = r

        self.glora_A_k = nn.Linear(dim, r, bias=False)
        self.glora_B_k = nn.Linear(r, dim, bias=False)
        self.glora_A_v = nn.Linear(dim, r, bias=False)
        self.glora_B_v = nn.Linear(r, dim, bias=False)
        self.rank = r

        self.matrix = torch.zeros(dim ,dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(dim ,dim)
        self.n_cur_matrix = 0

    def init_param(self):
        nn.init.kaiming_uniform_(self.glora_A_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.glora_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.glora_B_k.weight)
        nn.init.zeros_(self.glora_B_v.weight)

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, task, register_hook=False, get_feat=False,get_cur_feat=False):
        # x size [128,197,768]
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_matrix + x.shape[0]*x.shape[1])
            self.n_matrix += x.shape[0]*x.shape[1]
        if get_cur_feat:
            # average self-attention
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0]*x.shape[1])
            self.n_cur_matrix += x.shape[0]*x.shape[1]

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # insert lora
        weight_k = torch.mm(self.glora_B_k.weight, self.glora_A_k.weight)
        weight_v = torch.mm(self.glora_B_v.weight, self.glora_A_v.weight)
        k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention_HLoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.rank = r

        self.glora_A_k = nn.Linear(dim, r, bias=False)
        self.elora_B_k = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.glora_A_v = nn.Linear(dim, r, bias=False)
        self.elora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])

        self.matrix = torch.zeros(dim ,dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(dim ,dim)
        self.n_cur_matrix = 0

    def init_param(self):
        nn.init.kaiming_uniform_(self.glora_A_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.glora_A_v.weight, a=math.sqrt(5))
        for t in range(len(self.elora_B_k)):
            nn.init.zeros_(self.elora_B_k[t].weight)
            nn.init.zeros_(self.elora_B_v[t].weight)

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, task, register_hook=False, get_feat=False,get_cur_feat=False):
        # x size [128,197,768]
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_matrix + x.shape[0]*x.shape[1])
            self.n_matrix += x.shape[0]*x.shape[1]
        if get_cur_feat:
            # average self-attention
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0]*x.shape[1])
            self.n_cur_matrix += x.shape[0]*x.shape[1]

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # insert lora
        # W + A(sum B)h
        weight_k = torch.mm(torch.stack([self.elora_B_k[t].weight for t in range(task+1)], dim=0).sum(dim=0), self.glora_A_k.weight)
        weight_v = torch.mm(torch.stack([self.elora_B_v[t].weight for t in range(task+1)], dim=0).sum(dim=0), self.glora_A_v.weight)
        k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _AjtoStats(nn.Module):
    def __init__(self, D, E=None, ridge_lambda=1e-9):
        super().__init__()
        E = D if E is None else E
        self.F = torch.zeros(D, D)
        self._F_acc = torch.zeros(D, D)

        self.weight = torch.zeros(E, D)
        self.ridge_lambda = ridge_lambda
        self.collect = False
        self.use_theta = False

    @torch.no_grad()
    def begin(self, ridge_lambda=None):
        if ridge_lambda is not None:
            self.ridge_lambda = float(ridge_lambda)
        self.collect = True
        self._F_acc.zero_()


    @torch.no_grad()
    def end(self):
        self.collect = False
        self.use_theta = False

    @torch.no_grad()
    def accumulate_F(self, X_flat):  # X_flat: (BN)×D
        if self.collect:

            self._F_acc = X_flat.to(self.F.device)


    @torch.no_grad()
    def end_and_solve(self, task,deltaW):  # deltaW: E×D

        device = deltaW.device

        prev = self.F.to(device=device)
        cur = self._F_acc.to(device=device)
        m = prev + cur

        m_inv = torch.linalg.inv(torch.eye(m.shape[0], device=device) * self.ridge_lambda + m)  # [D, D]
        prev_inv = torch.linalg.inv(torch.eye(prev.shape[0], device=device) * self.ridge_lambda + prev)
        if task > 0 :


            deltaW =  deltaW/(task+1)


        self.weight.copy_(deltaW)


        self.F = m

        self.collect = False
        self.use_theta = True
