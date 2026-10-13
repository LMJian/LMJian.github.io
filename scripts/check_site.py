#!/usr/bin/env python3
"""Check generated HTML resources, internal anchors, feed and search data."""
import json
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = set()
        self.refs = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        for key in ("href", "src", "data-src"):
            if attrs.get(key):
                self.refs.append(attrs[key])


def check(root, baseurl=""):
    pages = {p: Page(p.read_text()) for p in root.rglob("*.html")}
    errors = []
    checked = 0
    for path, page in pages.items():
        current = "/" + path.relative_to(root).as_posix()
        if current.endswith("index.html"):
            current = current[:-10]
        current = baseurl + current
        for ref in page.refs:
            parts = urlsplit(ref)
            if parts.scheme or parts.netloc or ref == "#":
                continue
            target = urlsplit(urljoin(current, ref))
            pathname = unquote(target.path)
            if baseurl and not pathname.startswith(baseurl + "/"):
                errors.append(f"{path.relative_to(root)}: outside baseurl: {ref}")
                continue
            filename = root / pathname[len(baseurl):].lstrip("/")
            if filename.is_dir():
                filename /= "index.html"
            if not filename.is_file():
                errors.append(f"{path.relative_to(root)}: missing: {ref}")
            elif target.fragment and filename in pages:
                if unquote(target.fragment) not in pages[filename].ids:
                    errors.append(f"{path.relative_to(root)}: missing anchor: {ref}")
            checked += 1
    ET.parse(root / "feed.xml")
    ET.parse(root / "sitemap.xml")
    search = json.loads((root / "assets/js/data/search.json").read_text())
    if not search:
        errors.append("Search index contains no posts")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"PASS: {len(pages)} HTML pages, {checked} local references, feed, sitemap and search JSON")


if __name__ == "__main__":
    check(Path(sys.argv[1] if len(sys.argv) > 1 else "_site").resolve(),
          sys.argv[2].rstrip("/") if len(sys.argv) > 2 else "")
