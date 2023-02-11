# https://velog.io/@aqaqsubin/Transformer-Attention-Is-All-You-Need#position-wise-feed-forward-network
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.ones(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return x

class ResidualConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        y = self.norm(x)
        y = sublayer(y)
        y = self.dropout(y)
        x = x + y
        return x

class GELU(nn.Module):
    def forward(self, x):
        gelu = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return gelu

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation=GELU()):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x

def clones(module, N):
    "N개의 동일한 모듈을 복사해 생성"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
