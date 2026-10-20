"""Synthetic positive house prices. Educational data, not a real market model."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
torch.set_num_threads(2)
rng = np.random.default_rng(42)
n = 1200
area = rng.uniform(40, 200, n)
age = rng.uniform(0, 40, n)
district = rng.choice(["A", "B", "C"], n)
effect = pd.Series(district).map({"A": 0.25, "B": 0.0, "C": -0.2}).to_numpy()
price = np.exp(12 + 0.008 * area - 0.006 * age + effect + rng.normal(0, 0.08, n))
frame = pd.DataFrame({"area": area, "age": age, "district": district})
frame.loc[rng.random(n) < 0.05, "age"] = np.nan
frame.loc[rng.random(n) < 0.05, "district"] = np.nan
train_ids, test_ids = train_test_split(np.arange(n), test_size=0.2, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.25, random_state=43)

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("fill", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
    ]), ["area", "age"]),
    ("category", Pipeline([
        ("fill", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), ["district"]),
])
x_train = preprocessor.fit_transform(frame.iloc[train_ids])
x_val = preprocessor.transform(frame.iloc[val_ids])
x_test = preprocessor.transform(frame.iloc[test_ids])
log_price = np.log(price)
target_mean = log_price[train_ids].mean()
target_std = log_price[train_ids].std()


def make_tensors(features, ids):
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor((log_price[ids] - target_mean) / target_std,
                     dtype=torch.float32).unsqueeze(1)
    return x, y


xt, yt = make_tensors(x_train, train_ids)
xv, yv = make_tensors(x_val, val_ids)
xe, ye = make_tensors(x_test, test_ids)
loader = DataLoader(TensorDataset(xt, yt), batch_size=64, shuffle=True)
model = nn.Sequential(
    nn.Linear(xt.shape[1], 32), nn.BatchNorm1d(32),
    nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1),
)
nn.init.kaiming_normal_(model[0].weight, nonlinearity="relu")
nn.init.xavier_normal_(model[-1].weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
loss_fn = nn.MSELoss()
best_loss, best_state = float("inf"), None
history = []
for epoch in range(80):
    model.train()
    total = 0.0
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    model.eval()
    with torch.inference_mode():
        val_loss = loss_fn(model(xv), yv).item()
    history.append({"epoch": epoch + 1, "train_loss": total / len(xt), "val_loss": val_loss})
    if val_loss < best_loss:
        best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
model.eval()
with torch.inference_mode():
    predicted_log = model(xe).squeeze(1).numpy() * target_std + target_mean
predicted_price = np.exp(predicted_log)
result = {
    "data": "synthetic, not real market prices",
    "train_samples": len(xt), "val_samples": len(xv), "test_samples": len(xe),
    "features": preprocessor.get_feature_names_out().tolist(),
    "test_log_rmse": float(np.sqrt(np.mean((predicted_log - log_price[test_ids]) ** 2))),
    "test_mae": float(np.mean(np.abs(predicted_price - price[test_ids]))),
    "constant_baseline_log_rmse": float(np.sqrt(np.mean((target_mean - log_price[test_ids]) ** 2))),
}
print(json.dumps(result, indent=2))
out = Path("runs/house")
out.mkdir(parents=True, exist_ok=True)
(out / "metrics.json").write_text(json.dumps(result, indent=2))
(out / "history.json").write_text(json.dumps(history, indent=2))
