import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
torch.set_num_threads(2)
x = torch.randn(400, 1)
y = 2.5 * x + 5.2 + 0.1 * torch.randn(400, 1)
x_train, x_test = x[:320], x[320:]
y_train, y_test = y[:320], y[320:]
loader = DataLoader(
    TensorDataset(x_train, y_train), batch_size=32, shuffle=True
)
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
history = []

for epoch in range(80):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xb)
        loss = loss_fn(prediction, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    history.append(total_loss / len(x_train))

model.eval()
with torch.inference_mode():
    test_mse = loss_fn(model(x_test), y_test).item()
result = {
    "weight": model.weight.item(),
    "bias": model.bias.item(),
    "first_train_mse": history[0],
    "last_train_mse": history[-1],
    "test_mse": test_mse,
}
print(json.dumps(result, indent=2))
out = Path("runs/linear")
out.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), out / "model.pt")
(out / "metrics.json").write_text(json.dumps(result, indent=2))
(out / "history.json").write_text(json.dumps(history))

# Verify that persistence preserves predictions.
restored = nn.Linear(1, 1)
restored.load_state_dict(torch.load(out / "model.pt", weights_only=True))
restored.eval()
with torch.inference_mode():
    torch.testing.assert_close(restored(x_test), model(x_test))
