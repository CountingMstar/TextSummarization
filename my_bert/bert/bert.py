import torch.nn as nn
from transformer.transformer import TransformerEncoder
from transformer.embedding import BertEmbedding
from transformer.utils import clones

class BERT(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=12, attention_heads=12, dropout=0.1):
        super(BERT, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attention_heads = attention_heads
        self.dropout = dropout

        self.feed_forward_hidden = hidden * 4
        self.embedding = BertEmbedding(vocab_size, embed_size=hidden, dropout=dropout)
        self.transformer_encoder_blocks = clones(TransformerEncoder(head_num=attention_heads, d_model=hidden, d_ff=self.feed_forward_hidden, dropout=dropout), n_layers)

    def forward(self, x, segment_label):
        # True, False making 
        # ex) x = tensor[1, 2, 0], (x > 0) == [True, True, False]
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x, segment_label)
        
        for transformer_encoder in self.transformer_encoder_blocks:
            x = transformer_encoder.forward(x, mask)
        return x

