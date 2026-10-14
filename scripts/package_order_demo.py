#!/usr/bin/env python3
"""按固定清单生成博客示例 ZIP，不打包模型、缓存或本地环境。"""
import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/order-cancel"
TARGET = ROOT / "assets/downloads/order-cancel-demo.zip"
FILES = (
    ".gitignore", "README.md", "requirements.txt", "orders.csv",
    "generate_data.py", "train.py", "predict.py", "app.py", "test_demo.py",
)


def package():
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name in FILES:
            info = ZipInfo("order-cancel-demo/" + name, date_time=(2026, 10, 14, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (SOURCE / name).read_bytes())
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = package()
    if args.check:
        if not TARGET.exists() or TARGET.read_bytes() != payload:
            raise SystemExit("示例 ZIP 与源码不一致，请运行 python3 scripts/package_order_demo.py")
        print(f"PASS: 下载包与 {len(FILES)} 个源文件一致")
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_bytes(payload)
        print(f"写入 {TARGET.relative_to(ROOT)} ({len(payload)} bytes)")


if __name__ == "__main__":
    main()
