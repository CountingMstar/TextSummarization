import torch
import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super(SegmentEmbedding, self).__init__(3, embed_size, padding_idx=0)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super(TokenEmbedding, self).__init__(vocab_size, embed_size, padding_idx=0)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pe.requires_grad = False
        position = torch.arrange(0, max_len).unsqueeze(1)
        base = torch.ones(d_model // 2).fill_(10000)
        pow_term = torch.arange(0, d_model, 2) / torch.tensor(d_model, dtype=torch.float32)
        div_term = torch.pow(base, pow_term)

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.pe[:, :x.size(1)]
        return x

class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super(BertEmbedding, self).__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.embed_size = embed_size

    def forword(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        x = self.dropout(x)
        return x