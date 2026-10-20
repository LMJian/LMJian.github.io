"""MNIST + MLP, or Fashion-MNIST + CNN. Data is downloaded on the first run."""
import argparse
import copy
import json
import socket
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def build_model(task):
    if task == "mnist":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 50), nn.ReLU(),
            nn.Linear(50, 100), nn.ReLU(),
            nn.Linear(100, 10),
        )
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 7 * 7, 10),
    )


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += loss_fn(logits, y).item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            count += len(y)
    return {"loss": total_loss / count, "accuracy": correct / count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["mnist", "fashion"], default="mnist")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit", type=int, default=12000,
                        help="Training pool size; 0 uses all 60,000 images")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="runs")
    args = parser.parse_args()
    if args.epochs < 1 or args.limit < 0 or 0 < args.limit < 20:
        parser.error("epochs must be positive; limit must be 0 or at least 20")
    socket.setdefaulttimeout(30)  # Fail visibly instead of waiting indefinitely on a mirror.
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    dataset_type = datasets.MNIST if args.task == "mnist" else datasets.FashionMNIST
    # HTTPS mirrors; torchvision still verifies the dataset's built-in checksums.
    dataset_type.mirrors = (
        ["https://storage.googleapis.com/cvdf-datasets/mnist/"]
        if args.task == "mnist" else
        ["https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"]
    )
    transform = transforms.ToTensor()  # uint8 HWC -> float32 CHW, divide by 255
    pool = dataset_type(args.data_dir, train=True, download=True, transform=transform)
    test = dataset_type(args.data_dir, train=False, download=True, transform=transform)
    if args.limit:
        ids = torch.randperm(len(pool), generator=torch.Generator().manual_seed(42))
        pool = Subset(pool, ids[:min(args.limit, len(pool))].tolist())
    n_val = max(1, len(pool) // 5)
    train, val = random_split(
        pool, [len(pool) - n_val, n_val],
        generator=torch.Generator().manual_seed(43),
    )
    train_loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=256, num_workers=0)
    test_loader = DataLoader(test, batch_size=256, num_workers=0)
    model = build_model(args.task).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss, best_state, history = float("inf"), None, []

    for epoch in range(args.epochs):
        model.train()
        total_loss, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            count += len(y)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        row = {"epoch": epoch + 1, "train_loss": total_loss / count,
               "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    out = Path(args.output_dir) / args.task
    out.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state, out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    metadata = {
        "task": args.task, "epochs": args.epochs, "device": str(device),
        "train_samples": len(train), "val_samples": len(val), "test_samples": len(test),
        "torch": torch.__version__, "best_val_loss": best_loss,
        "test": test_metrics,
        "preprocessing": "ToTensor: grayscale uint8 -> [N,1,28,28] float32 / 255",
    }
    (out / "metrics.json").write_text(json.dumps(metadata, indent=2))
    restored = build_model(args.task)
    restored.load_state_dict(torch.load(out / "model.pt", map_location="cpu", weights_only=True))
    restored.eval()
    model = model.cpu().eval()
    x, y = next(iter(test_loader))
    with torch.inference_mode():
        expected = model(x)
        actual = restored(x)
        torch.testing.assert_close(actual, expected)
        probability = actual.softmax(dim=1)
    print(json.dumps({"test": test_metrics, "reload_matches": True,
                      "first_prediction": actual[0].argmax().item(),
                      "first_label": y[0].item(),
                      "first_max_probability": probability[0].max().item()}, indent=2))


if __name__ == "__main__":
    main()
