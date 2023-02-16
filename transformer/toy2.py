# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
# from torchtext.datasets import multi30k, Multi30k
# from typing import Iterable, List

import torch

import torchtext
from torchtext.legacy.datasets import Multi30k

# print(torch.__version__)
# print(torchtext.__version__)
a = torch.rand(5).cuda()
print(a)


# print(Multi30k)

# import torch
# import torchtext.legacy.data as data
# from torchtext.legacy.datasets import Multi30k

# SRC = data.Field(
#     tokenize="spacy",
#     tokenizer_language="en",
#     init_token="<sos>",
#     eos_token="<eos>",
#     lower=True,
# )

# TRG = data.Field(
#     tokenize="spacy",
#     tokenizer_language="de",
#     init_token="<sos>",
#     eos_token="<eos>",
#     lower=True,
# )

# print("#######yes#######")
