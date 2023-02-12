import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ResidualConnection, FeedForward, GELU, clones
from attention import MultiHeadAttention
from embedding import PositionalEmbedding
import math


class TransformerEncoder(nn.Module):
    def __init__(self, head_num, d_model, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.multi_head_attention = MultiHeadAttention(head_num, d_model, dropout=dropout)
        self.residual_1 = ResidualConnection(d_model, dropout=dropout)
        self.residual_2 = ResidualConnection(d_model, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=0.1, activation=GELU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, mask):
        x = self.residual_1(input, lambda x: self.multi_head_attention(x, x, x, mask))
        x = self.residual_2(x, lambda x: self.feed_forward(x))
        x = self.dropout(x) 
        return x


#############################################
# 여기서부터는 BERT에서는 불필요. Transformer 구조.
#############################################
class TransformerDecoder(nn.Module):
    def __init__(self, head_num, d_model, d_ff, dropout):
        super(TransformerDecoder,self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(head_num, d_model, dropout=dropout)
        self.residual_1 = ResidualConnection(d_model, dropout=dropout)

        self.encoder_decoder_attention = MultiHeadAttention(head_num, d_model, dropout=dropout)
        self.residual_2 = ResidualConnection(d_model, dropout=dropout)

        self.feed_forward= FeedForward(d_model, d_ff, dropout=0.1, activation=GELU())
        self.residual_3 = ResidualConnection(d_model, dropout=dropout)

    def forward(self, target, encoder_output, target_mask, encoder_mask):
        # target, x, target_mask, input_mask
        x = self.residual_1(target, lambda x: self.masked_multi_head_attention(x, x, x, target_mask))
        x = self.residual_2(x, lambda x: self.encoder_decoder_attention(x, encoder_output, encoder_output, encoder_mask))
        x = self.residual_3(x, self.feed_forward)
        return x

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class Embeddings(nn.Module):
    def __init__(self, vocab_num, d_model):
            super(Embeddings,self).__init__()
            self.emb = nn.Embedding(vocab_num,d_model)
            self.d_model = d_model
    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)

class Transformer(nn.Module):
    def __init__(self,input_vocab_num, output_vocab_num, d_model, max_seq_len, head_num, dropout, N):
        super(Transformer,self).__init__()
        self.input_embedding = Embeddings(input_vocab_num, d_model)
        self.output_embedding = Embeddings(output_vocab_num, d_model)
        self.positional_encoding = PositionalEmbedding(d_model, max_seq_len, dropout)

        self.encoders = clones(TransformerEncoder(head_num, d_model, d_ff, dropout), N)
        self.decoders = clones(TransformerDecoder(head_num, d_model, d_ff, dropout), N)

        self.generator = Generator(d_model, output_vocab_num)

    def forward(self, input, target, input_mask, target_mask, labels=None):
        x = self.positional_encoding(self.input_embedding(input))
        for encoder in self.encoders:
            x = encoder(x, input_mask)

        target = self.positional_encoding(self.output_embedding(target))
        for decoder in self.decoders:
            # target, encoder_output, target_mask, encoder_mask)
            target = decoder(target, x, target_mask, input_mask)

        lm_logits = self.generator(target)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=3)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return lm_logits, loss

    def encode(self,input, input_mask):
        x = self.positional_encoding(self.input_embedding(input))
        for encoder in self.encoders:
            x = encoder(x, input_mask)
        return x

    def decode(self, encode_output, encoder_mask, target, target_mask):
        target = self.positional_encoding(self.output_embedding(target))
        for decoder in self.decoders:
        #target, encoder_output, target_mask, encoder_mask
            target = decoder(target, encode_output, target_mask, encoder_mask)
            lm_logits = self.generator(target)
        return lm_logits