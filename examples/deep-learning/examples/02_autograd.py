import torch

x = torch.tensor(2.0)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(10.0)
prediction = w * x + b
loss = (prediction - y).square()
loss.backward()
print("prediction / loss:", prediction.item(), loss.item())
print("dw / db:", w.grad.item(), b.grad.item())
assert w.grad.item() == -12.0 and b.grad.item() == -6.0

# backward only computes gradients; an optimizer changes parameters.
optimizer = torch.optim.SGD([w, b], lr=0.1)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
print("updated w / b:", round(w.item(), 2), round(b.item(), 2))

u = torch.tensor(2.0, requires_grad=True)
v = (u * u).detach()
(v * u).backward()
print("detached gradient:", u.grad.item())  # v is treated as a constant: 4
assert u.grad.item() == 4.0
