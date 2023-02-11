import copy
import torch
import torch.nn as nn
from utils import clones, ResidualConnection, FeedForward, GELU
from attention import MultiHeadAttention


class Encoder(nn.Module):
    def __init__(self, head_num, d_model, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.multi_head_attention = MultiHeadAttention(head_num, d_model, dropout=dropout)
        self.residual_1 = ResidualConnection(d_model, dropout=dropout)
        self.residual_2 = ResidualConnection(d_model, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=0.1, activation=GELU())

    def forward(self, input, mask):
        x = self.residual_1(input, lambda x: self.multi_head_attention(x, x, x, mask))
        x = self.residual_2(x, lambda x: self.feed_forward(x))