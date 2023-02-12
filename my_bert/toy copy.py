import torch

x = torch.tensor([[ 3,  4,  4,  9,  2, 16,  5,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0],
        [ 3,  6, 14,  4,  2, 13,  5,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0],
        [ 3, 10,  4,  2, 16,  5,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0]])


mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1)

print(mask)