# GRU模型
import torch
import torch.nn as nn
rnn = nn.GRU(5, 6, 2)
input = torch.randn(1, 3, 5)
h0 = torch.randn(2, 3, 6)
output, hn = rnn(input, h0)
print(output)
print(output.shape)
print(hn)
print(hn.shape)