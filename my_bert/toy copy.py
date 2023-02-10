import torch

a = torch.randn(3, 4)

a.transpose_(0, 1)

b = torch.randn(4, 3)

# 두 tensor는 모두 (4, 3) shape
print(a)
'''
tensor([[-0.7290,  0.7509,  1.1666],
        [-0.9321, -0.4360, -0.2715],
        [ 0.1232, -0.6812, -0.0358],
        [ 1.1923, -0.8931, -0.1995]])'''

print(b)
'''
tensor([[-0.1630,  0.1704,  1.8583],
        [-0.1231, -1.5241,  0.2243],
        [-1.3705,  1.2717, -0.6051],
        [ 0.0412,  1.3312, -1.2066]])'''

# a 텐서 메모리 주소 예시
for i in range(4):
    for j in range(3):
        print(a[i][j].data_ptr())
'''
94418119497152
94418119497168
94418119497184
94418119497156
94418119497172
94418119497188
94418119497160
94418119497176
94418119497192
94418119497164
94418119497180
94418119497196'''
print()
# b 텐서 메모리 주소 예시
for i in range(4):
    for j in range(3):
        print(b[i][j].data_ptr())
'''
94418119613696
94418119613700
94418119613704
94418119613708
94418119613712
94418119613716
94418119613720
94418119613724
94418119613728
94418119613732
94418119613736
94418119613740'''
print()
print(a.stride()) # (1, 4)
print(b.stride()) # (3, 1)


print(a.is_contiguous()) # False
print(b.is_contiguous()) # True

a = a.contiguous()
print(a.is_contiguous())
for i in range(4):
    for j in range(3):
        print(a[i][j].data_ptr())

print('##############')
print(b)
print(b.is_contiguous())
b = b.transpose(-2, 1)
print(b)
print(b.is_contiguous())
for i in range(3):
    for j in range(4):
        print(b[i][j].data_ptr())