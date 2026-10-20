import numpy as np
import torch

torch.manual_seed(42)
x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
w = torch.ones(3, 2)
print("x:", x.tolist())
print("elementwise:", (x * x).tolist())
print("matrix product:", (x @ w).tolist())
print("column mean:", x.mean(dim=0).tolist())
print("row mean:", x.mean(dim=1, keepdim=True).tolist())
print("paired indexing:", x[[0, 1], [1, 2]].tolist())
print("boolean indexing:", x[x > 2].tolist())
print("transpose:", tuple(x.transpose(0, 1).shape))
print("cat:", tuple(torch.cat([x, x], dim=0).shape))
print("stack:", tuple(torch.stack([x, x], dim=0).shape))
print("unsqueeze:", tuple(x.unsqueeze(1).shape))

a = np.array([1.0, 2.0], dtype=np.float32)
shared = torch.from_numpy(a)
copied = torch.tensor(a)
a[0] = 9.0
print("shared / copied:", shared.tolist(), copied.tolist())
assert shared[0].item() == 9.0 and copied[0].item() == 1.0

t = x.transpose(0, 1)
print("contiguous:", t.is_contiguous())
print("reshape:", t.reshape(-1).tolist())
print("contiguous then view:", t.contiguous().view(-1).tolist())
assert torch.equal(t.reshape(-1), t.contiguous().view(-1))
