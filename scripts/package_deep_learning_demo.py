#!/usr/bin/env python3
"""Build the deep-learning download from a fixed, reproducible manifest."""
import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/deep-learning"
IMAGES = ROOT / "assets/img/posts/deep-learning"
TARGET = ROOT / "assets/downloads/deep-learning-demo.zip"
EXAMPLE_FILES = (
    ".gitignore", "README.md",
    "examples/01_tensor_basics.py",
    "examples/02_autograd.py",
    "examples/03_linear_regression.py",
    "examples/04_vision.py",
    "examples/05_house_prices.py",
    "examples/requirements.txt",
    "examples/expected-results.json",
)
IMAGE_FILES = (
    "01-learning-system.png", "02-tensor-shapes.png", "03-activations.png",
    "04-regression-losses.png", "05-backpropagation.png",
    "06-linear-training.png", "07-optimization.png", "08-generalization.png",
    "09-convolution.png", "10-pooling.png", "11-cnn-shapes.png",
    "12-predictions.png", "13-dense-forward.png", "14-bn-dropout.png",
    "15-padding-stride.png", "16-multichannel.png",
    "17-inception-residual.png", "18-activation-variants.png",
    "README.md", "fashion-mnist-LICENSE.txt",
)


def package():
    files = [(name, SOURCE / name) for name in EXAMPLE_FILES]
    files += [("assets/" + name, IMAGES / name) for name in IMAGE_FILES]
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, source in files:
            info = ZipInfo("deep-learning-demo/" + name, (2026, 10, 20, 0, 0, 0))
            info.create_system = 3
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = package()
    if args.check:
        if not TARGET.exists() or TARGET.read_bytes() != payload:
            raise SystemExit(
                "下载包与源码或图片不一致，请运行 "
                "python3 scripts/package_deep_learning_demo.py"
            )
        print(f"PASS: 深度学习下载包与 {len(EXAMPLE_FILES) + len(IMAGE_FILES)} 个源文件一致")
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_bytes(payload)
        print(f"写入 {TARGET.relative_to(ROOT)} ({len(payload)} bytes)")


if __name__ == "__main__":
    main()
