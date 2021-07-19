import torch

# 一
x = torch.tensor([1, 2, 3, 4])
print(x)
print(x.shape)

y=torch.unsqueeze(x, 0)
print(y)
print(y.shape)

z=torch.unsqueeze(x, 1)
print(z)
print(z.shape)

# 二
a = torch.randn(4)
print(a)
b = torch.randn(4, 1)
print(b)
print("------------")
c=torch.add(a, b)
print(c)
d=torch.add(a, b, alpha=10)         # 意思是b*alpha + a
print(d)