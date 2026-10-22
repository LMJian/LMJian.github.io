#!/usr/bin/env python3
"""Build the Jev article, figures and minimal examples from a fixed manifest."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/jev"
POST = ROOT / "_posts/2026-10-23-jev-for-backend-engineers.md"
IMAGES = ROOT / "assets/img/posts/jev"
TARGET = ROOT / "assets/downloads/jev-demo.zip"
ARTICLE = "jev-for-backend-engineers.md"
SOURCE_FILES = (
    ".gitignore", "README.md", "review.md",
    "examples/decision_demo.py", "examples/request.json", "examples/call_jev.sh",
)
IMAGE_FILES = (
    "01-decision-path.png", "02-calibration.png", "03-agent-architecture.png",
)
WEB_EXAMPLE = "均收在[完整学习包](#download)中"
OFFLINE_EXAMPLE = "与本文一同提供"


def offline_article():
    header, separator, body = POST.read_text(encoding="utf-8").partition("\n---\n")
    if not separator:
        raise ValueError("文章缺少 YAML front matter")
    title = json.loads(re.search(r"^title: (.+)$", header, re.M)[1])
    body, count = re.subn(
        r"<!-- download-package:start -->.*?<!-- download-package:end -->\s*",
        "", body, flags=re.S,
    )
    if count != 1:
        raise ValueError("文章必须包含一个下载包说明块")
    pattern = r"!\[([^\]]*)\]\(([^)]+\.png)\)\{:[^}]+\}"
    if sorted(match[1] for match in re.findall(pattern, body)) != sorted(IMAGE_FILES):
        raise ValueError("正文图片与固定打包清单不一致")
    body = re.sub(pattern, r"![\1](assets/\2)", body)
    if body.count(WEB_EXAMPLE) != 1:
        raise ValueError("示例下载说明已变化，请同步离线转换规则")
    body = body.replace(WEB_EXAMPLE, OFFLINE_EXAMPLE)
    if "{{" in body or "{%" in body:
        raise ValueError("离线正文中仍有未处理的 Liquid 模板")
    return ("# " + title + "\n\n" + body.lstrip()).encode("utf-8")


def source_bytes(name):
    path = SOURCE / name
    if path.suffix != ".md":
        return path.read_bytes()
    text = path.read_text(encoding="utf-8")
    text = text.replace("../../_posts/" + POST.name, ARTICLE)
    text = text.replace("../../assets/img/posts/jev/", "assets/")
    return text.encode("utf-8")


def package():
    files = [(ARTICLE, offline_article())]
    files += [(name, source_bytes(name)) for name in SOURCE_FILES]
    files += [("assets/" + name, (IMAGES / name).read_bytes()) for name in IMAGE_FILES]
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, payload in files:
            info = ZipInfo("jev-demo/" + name, (2026, 10, 23, 0, 0, 0))
            info.create_system = 3
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = package()
    if args.check:
        if not TARGET.is_file() or TARGET.read_bytes() != payload:
            raise SystemExit(
                "下载包与文章、源码或图片不一致，请运行 "
                "python3 scripts/package_jev_demo.py"
            )
        print(f"PASS: Jev 下载包与 {1 + len(SOURCE_FILES) + len(IMAGE_FILES)} 个源文件一致")
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_bytes(payload)
        print(f"写入 {TARGET.relative_to(ROOT)} ({len(payload)} bytes)")


if __name__ == "__main__":
    main()
