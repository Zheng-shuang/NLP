# 导入工具包
import torch
import torch.nn as nn
# input_size: 输入张量x中特征维度的大小,hidden_size: 隐层张量h中特征维度的大小,num_layers: 隐含层的数量,nonlinearity: 激活函数的选择, 默认是tanh.
rnn = nn.RNN(5, 6, 1)

# 设定输入的张量x
# 1表示输入序列的长度，3表示batch_size（批次的样本数），5表示输入张量x中特征维度的大小，与上面的5要一致
input = torch.randn(1, 3, 5)

# 设定初始化的h0
# 1表示num_layers*num_directions(层数*网络方向数，单向表示1，双向表示2），和最上面的1对应上，
# 3表示batch_size（批次的样本数），6表示hidden_size: 隐层张量h中特征维度的大小
h0 = torch.randn(1, 3, 6)

output, hn = rnn(input, h0)
print(output)
print(output.shape)          # 展示维度
print(hn)
print(hn.shape)