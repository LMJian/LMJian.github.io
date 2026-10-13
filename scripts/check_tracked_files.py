#!/usr/bin/env python3
"""Reject tracked or staged files that match Git ignore rules."""
import json
import subprocess
from pathlib import Path


def check(root):
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--ignored", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    paths = [path.decode("utf-8", errors="replace")
             for path in result.stdout.split(b"\0") if path]
    if paths:
        listed = "\n".join(f"  {json.dumps(path, ensure_ascii=False)}" for path in paths)
        raise SystemExit(
            "FAIL: ignored files are still tracked or staged:\n"
            f"{listed}\n"
            "Remove their Git tracking with git rm --cached -- <path> "
            "to keep the local files."
        )
    print("PASS: no tracked or staged files match Git ignore rules")


if __name__ == "__main__":
    check(Path(__file__).resolve().parents[1])
