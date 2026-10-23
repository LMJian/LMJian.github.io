#!/usr/bin/env python3
"""Build and check the prompt-engineering article, figures and minimal example."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/prompt-engineering"
POST = ROOT / "_posts/2026-10-23-prompt-engineering-for-backend-engineers.md"
IMAGES = ROOT / "assets/img/posts/prompt-engineering"
TARGET = ROOT / "assets/downloads/prompt-engineering-demo.zip"
ARTICLE = "prompt-engineering-for-backend-engineers.md"
SOURCE_FILES = (
    "README.md", "review.md", "requirements.txt",
    "examples/validate_triage.py", "examples/triage.schema.json",
)
IMAGE_FILES = (
    "01-task-contract.png", "02-validation-layers.png", "03-instruction-boundary.png",
)
WEB_EXAMPLE = "可运行的[最小示例](#download)和[JSON Schema](#download)均收在完整学习包中，Schema 由同一个数据模型生成。"
OFFLINE_EXAMPLE = "同目录的[最小示例](examples/validate_triage.py)可以直接运行，[JSON Schema](examples/triage.schema.json)由同一个数据模型生成。"


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
    body = body.replace("下载学习包并解压后，在离线 Markdown 所在目录执行：", "在本文所在目录执行：")
    if "{{" in body or "{%" in body:
        raise ValueError("离线正文中仍有未处理的 Liquid 模板")
    return ("# " + title + "\n\n" + body.lstrip()).encode("utf-8")


def source_bytes(name):
    path = SOURCE / name
    if path.suffix != ".md":
        return path.read_bytes()
    text = path.read_text(encoding="utf-8")
    text = text.replace("../../_posts/" + POST.name, ARTICLE)
    text = text.replace("../../assets/img/posts/prompt-engineering/", "assets/")
    return text.encode("utf-8")


def package():
    files = [(ARTICLE, offline_article())]
    files += [(name, source_bytes(name)) for name in SOURCE_FILES]
    files += [("assets/" + name, (IMAGES / name).read_bytes()) for name in IMAGE_FILES]
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, payload in files:
            info = ZipInfo("prompt-engineering-demo/" + name, (2026, 10, 23, 0, 0, 0))
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
            raise SystemExit("下载包与文章、源码或图片不一致，请运行 python3 scripts/package_prompt_engineering_demo.py")
        print(f"PASS: 提示词工程下载包与 {1 + len(SOURCE_FILES) + len(IMAGE_FILES)} 个源文件一致")
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_bytes(payload)
        print(f"写入 {TARGET.relative_to(ROOT)} ({len(payload)} bytes)")


if __name__ == "__main__":
    main()
