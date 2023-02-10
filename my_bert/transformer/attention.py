import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, dropout=0):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        K_T = torch.transpose(K, -2, -1)
        QK = torch.matmul(Q, K_T)
        d_k = K.size(-1)
        # attention score
        attention_score = QK/math.sqrt(d_k)

        # masking된 부분을 '0'으로 만듦
        # masked_fill: mask가 0인 부분을 -1e20로 채움('-큰수'가 softmax함수를 통과하면 '0'에 가깝게 됨)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e20)

        attention = F.softmax(attention_score, dim=-1)
        # dropout
        attention = self.dropout(attention)

        # 최종 attention값
        attention = torch.matmul(attention, V)

        return attention

class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % head_num == 0

        self.head_num = head_num
        self.d_model = d_model
        self.d_k = self.d_v = d_model // head_num

        self.Q_weigh = nn.Linear(d_model, d_model)
        self.K_weigh = nn.Linear(d_model, d_model)
        self.V_weigh = nn.Linear(d_model, d_model)
        self.O_weigh = nn.Linear(d_model, d_model)

        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, mask=None):
        # if mask is not None:
        #   # Same mask applied to all h heads.
        #   mask = mask.unsqueeze(1)

        batche_num = Q.size(0)

        # view: tensor의 내용은 건들지않고 모양만 변형
        Q = self.Q_weigh(Q).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)
        K = self.K_weigh(K).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)
        V = self.V_weigh(V).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)

        attention = self.attention(Q, K, V, mask)
        # contiguous(): tensor를 transpose, view하는 등 모향변환 과정을 거치면 메모리할당 순서가 바뀜. 이 순서를 다시 정렬해줌.
        attention = attention.transpose(1, 2).contiguous().view(batche_num, -1, self.head_num*self.d_k)
        attention = self.O_weigh(attention)

        return attention
