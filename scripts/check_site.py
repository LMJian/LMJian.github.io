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
        self.main_refs = []
        self.nav_icons = []
        self.in_main = False
        self.in_sidebar = False
        self.in_nav = False
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "main":
            self.in_main = True
        if tag == "aside" and attrs.get("id") == "sidebar":
            self.in_sidebar = True
        if tag == "nav" and self.in_sidebar:
            self.in_nav = True
        if tag == "i" and self.in_nav:
            self.nav_icons.append(set(attrs.get("class", "").split()))
        if self.in_main and tag == "a" and attrs.get("href"):
            self.main_refs.append(attrs["href"])
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        for key in ("href", "src", "data-src"):
            if attrs.get(key):
                self.refs.append(attrs[key])

    def handle_endtag(self, tag):
        if tag == "main":
            self.in_main = False
        elif tag == "aside":
            self.in_sidebar = False
        elif tag == "nav":
            self.in_nav = False


def check(root, baseurl=""):
    pages = {p: Page(p.read_text()) for p in root.rglob("*.html")}
    errors = []
    checked = 0
    for path, page in pages.items():
        icon_styles = {"fa", "fas", "far", "fab", "fa-solid", "fa-regular", "fa-brands"}
        if any(not classes & icon_styles for classes in page.nav_icons):
            errors.append(f"{path.relative_to(root)}: navigation icon missing font style")
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
    post_urls = {unquote(urlsplit(post["url"]).path) for post in search}
    for section in ("categories", "tags", "archives"):
        path = root / section / "index.html"
        if path not in pages:
            errors.append(f"Missing {section} index")
            continue
        main_urls = {unquote(urlsplit(ref).path) for ref in pages[path].main_refs}
        if section == "archives":
            if post_urls - main_urls:
                errors.append("Archives index is missing article links")
        elif any(post.get(section) for post in search):
            prefix = f"{baseurl}/{section}/"
            if not any(url.startswith(prefix) and url != prefix for url in main_urls):
                errors.append(f"{section} index has no links to its article lists")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"PASS: {len(pages)} HTML pages, {checked} local references, feed, sitemap and search JSON")


if __name__ == "__main__":
    check(Path(sys.argv[1] if len(sys.argv) > 1 else "_site").resolve(),
          sys.argv[2].rstrip("/") if len(sys.argv) > 2 else "")
