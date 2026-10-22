#!/usr/bin/env python3
"""Build an offline NLP article, examples and figures from a fixed manifest."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/nlp-huggingface"
IMAGES = ROOT / "assets/img/posts/nlp-huggingface"
POST = ROOT / "_posts/2026-10-22-nlp-huggingface-for-backend-engineers.md"
TARGET = ROOT / "assets/downloads/nlp-huggingface-demo.zip"
ARTICLE = "NLP与HuggingFace入门.md"
GITHUB_SOURCE = (
    "https://github.com/LMJian/LMJian.github.io/blob/main/"
    "examples/nlp-huggingface/"
)
EXAMPLE_FILES = (
    ".gitignore", "README.md", "coverage.md", "review.md",
    "examples/common.py", "examples/bow_baseline.py",
    "examples/tokenize_word2vec.py", "examples/attention_numpy.py",
    "examples/sequence_classifier.py", "examples/translation.py",
    "examples/hf_inspect.py", "examples/datasets_demo.py",
    "examples/hf_finetune.py", "examples/reviews.csv",
    "examples/requirements.txt", "examples/requirements-lock.txt",
)
IMAGE_FILES = (
    "01-text-to-model.png", "02-tokenization.png", "03-word2vec.png",
    "04-rnn-gates.png", "05-seq2seq-attention.png", "06-attention-math.png",
    "07-transformer.png", "08-train-generate-mask.png",
    "09-pretraining-objectives.png", "10-hf-workflow.png",
    "11-gated-state-flow.png", "12-multihead-attention.png",
    "13-position-encoding.png", "14-bert-input-fields.png",
)


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
    image_pattern = r"!\[([^\]]*)\]\(([^)]+\.png)\)\{:[^}]+\}"
    referenced = [match[1] for match in re.findall(image_pattern, body)]
    if sorted(referenced) != sorted(IMAGE_FILES):
        raise ValueError("正文图片与固定打包清单不一致")
    body = re.sub(image_pattern, r"![\1](assets/\2)", body)
    body = body.replace(GITHUB_SOURCE, "")
    body = body.replace("网页已启用数学公式渲染。下载包中的正文需要", "本文需要")
    if "{{" in body or "{%" in body:
        raise ValueError("离线正文中仍有未处理的 Liquid 模板")
    return ("# " + title + "\n\n" + body.lstrip()).encode("utf-8")


def source_bytes(name):
    path = SOURCE / name
    if path.suffix != ".md":
        return path.read_bytes()
    text = path.read_text(encoding="utf-8")
    text = text.replace("../../_posts/" + POST.name, ARTICLE)
    text = text.replace("../../assets/img/posts/nlp-huggingface/", "assets/")
    return text.encode("utf-8")


def package():
    files = [(ARTICLE, offline_article())]
    files += [(name, source_bytes(name)) for name in EXAMPLE_FILES]
    files += [("assets/" + name, (IMAGES / name).read_bytes()) for name in IMAGE_FILES]
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, payload in files:
            info = ZipInfo("nlp-huggingface-demo/" + name, (2026, 10, 22, 0, 0, 0))
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
        if not TARGET.exists() or TARGET.read_bytes() != payload:
            raise SystemExit(
                "下载包与文章、源码或图片不一致，请运行 "
                "python3 scripts/package_nlp_huggingface_demo.py"
            )
        print(f"PASS: NLP 下载包与 {1 + len(EXAMPLE_FILES) + len(IMAGE_FILES)} 个源文件一致")
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_bytes(payload)
        print(f"写入 {TARGET.relative_to(ROOT)} ({len(payload)} bytes)")


if __name__ == "__main__":
    main()
