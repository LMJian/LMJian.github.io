#!/usr/bin/env python3
"""Build/check the Personal Agent and REA learning packages from site sources."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "personal-agent": {
        "article": "personal-agent-for-backend-engineers.md",
        "images": ("agent-work-loop.png", "deployment-dimensions.png", "agent-permission-boundary.png"),
        "files": (".gitignore", "README.md", "review.md", "examples/task_state.py"),
        "links": {"task_state.py": "examples/task_state.py"},
    },
    "rea": {
        "article": "rea-for-backend-engineers.md",
        "images": ("rea-toolchain.png", "evidence-chain.png", "integration-checks.png"),
        "files": (
            ".gitignore", "README.md", "review.md", "examples/check-demo.mjs",
            "examples/owned-app/app.js", "examples/owned-app/package.json",
            "examples/codex-rea.toml", "examples/deepseek-rea.patch.yml",
        ),
        "links": {"app.js": "examples/owned-app/app.js", "check-demo.mjs": "examples/check-demo.mjs"},
    },
}


def post_path(config):
    return ROOT / "_posts" / ("2026-10-23-" + config["article"])


def offline_article(name, config):
    header, separator, body = post_path(config).read_text(encoding="utf-8").partition("\n---\n")
    if not separator:
        raise ValueError("Missing article front matter")
    title = json.loads(re.search(r"^title: (.+)$", header, re.M)[1])
    body, count = re.subn(
        r"<!-- download-package:start -->.*?<!-- download-package:end -->\s*",
        "", body, flags=re.S,
    )
    if count != 1:
        raise ValueError("Expected exactly one download block")
    pattern = r"!\[([^\]]*)\]\(([^)]+\.png)\)\{:[^}]+\}"
    if sorted(match[1] for match in re.findall(pattern, body)) != sorted(config["images"]):
        raise ValueError("Article images do not match the fixed manifest")
    body = re.sub(pattern, r"![\1](assets/\2)", body)
    for label, path in config["links"].items():
        source = f"[{label}](#download)"
        if body.count(source) != 1:
            raise ValueError(f"Expected one source-file link: {label}")
        body = body.replace(source, f"[{label}]({path})")
    body = body.replace(
        "下载学习包并解压后，在离线 Markdown 所在目录运行完整文件：",
        "在文章目录下运行完整文件：",
    )
    if "{{" in body or "{%" in body:
        raise ValueError("Unconverted Liquid markup in offline article")
    return ("# " + title + "\n\n" + body.lstrip()).encode("utf-8")


def package(name, config):
    source = ROOT / "examples" / name
    files = [(config["article"], offline_article(name, config))]
    for filename in config["files"]:
        payload = (source / filename).read_bytes()
        if filename.endswith(".md"):
            text = payload.decode("utf-8")
            text = text.replace("../../_posts/" + post_path(config).name, config["article"])
            text = text.replace(f"../../assets/img/posts/{name}/", "assets/")
            payload = text.encode("utf-8")
        files.append((filename, payload))
    files.extend(("assets/" + filename, (ROOT / "assets/img/posts" / name / filename).read_bytes())
                 for filename in config["images"])
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for filename, payload in files:
            info = ZipInfo(f"{name}-demo/{filename}", (2026, 10, 23, 0, 0, 0))
            info.create_system = 3
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return buffer.getvalue(), len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, config in PACKAGES.items():
        payload, count = package(name, config)
        target = ROOT / f"assets/downloads/{name}-demo.zip"
        if args.check:
            if not target.is_file() or target.read_bytes() != payload:
                raise SystemExit(
                    f"Stale {name} package; run python3 scripts/package_agent_learning_demos.py"
                )
            print(f"PASS: {name} ZIP matches {count} article/example/image files")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            print(f"Wrote {target.relative_to(ROOT)} ({len(payload)} bytes)")


if __name__ == "__main__":
    main()
