import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention_LoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rank = r


        self.lora_A_k = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_k = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])


        self.register_buffer("H_sum_k", torch.zeros(dim, dim))
        self.register_buffer("H_sum_v", torch.zeros(dim, dim))
        self.register_buffer("pre_k", torch.zeros(dim, dim))
        self.register_buffer("pre_v", torch.zeros(dim, dim))


        self.collect_fisher = False
        self._fisher_delta_k = None
        self._fisher_delta_v = None
        self.use_merged_weight = False



    def init_param(self):

        with torch.no_grad():

            for t in range(len(self.lora_B_k)):
                nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_k[t].weight)
                nn.init.zeros_(self.lora_B_v[t].weight)

    def forward(self, x, task):
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

        if self.use_merged_weight and not self.collect_fisher:
            cur_k = torch.zeros_like(pre_k)
            cur_v = torch.zeros_like(pre_v)
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

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_matrix(self, task):
        matrix_k = torch.mm(self.lora_B_k[task].weight, self.lora_A_k[task].weight)
        matrix_v = torch.mm(self.lora_B_v[task].weight, self.lora_A_v[task].weight)


        return matrix_k, matrix_v

    def start_task(self):
        """Activate the new task adapter on top of the accumulated weights."""
        self.use_merged_weight = False
