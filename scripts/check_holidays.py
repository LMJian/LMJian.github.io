#!/usr/bin/env python3
"""Validate the published holiday contract and the official 2026 date oracle."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

ROOT = Path(__file__).resolve().parents[1]


def validate(root):
    folder = root / "assets/holidays/cn"
    index_bytes = (folder / "index.json").read_bytes()
    assert len(index_bytes) <= 64 * 1024
    index = json.loads(index_bytes)
    assert index["schemaVersion"] == 1
    assert index["region"] == "CN"
    assert 1 <= len(index["years"]) <= 16
    years = set()
    for entry in index["years"]:
        year, revision = entry["year"], entry["revision"]
        assert type(year) is int and 2020 <= year <= 2100 and year not in years
        assert type(revision) is int and 1 <= revision <= 1_000_000
        years.add(year)
        assert entry["file"] == f"{year}.v{revision}.json"
        raw = (folder / entry["file"]).read_bytes()
        assert len(raw) <= 128 * 1024
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        data = json.loads(raw)
        assert (data["schemaVersion"], data["region"], data["year"], data["revision"]) == (1, "CN", year, revision)
        assert data["timeZone"] == "Asia/Shanghai"
        source = urlparse(data["source"]["url"])
        assert source.scheme == "https" and source.hostname and (source.hostname == "gov.cn" or source.hostname.endswith(".gov.cn"))
        assert 1 <= len(data["source"]["title"]) <= 200
        assert dt.date.fromisoformat(data["source"]["publishedAt"])
        seen = {}
        for item in data["overrides"]:
            date = dt.date.fromisoformat(item["date"])
            assert date.isoformat() == item["date"] and date.year == year
            assert item["date"] not in seen and item["kind"] in ("rest", "work")
            assert 1 <= len(item["name"]) <= 40
            seen[item["date"]] = item["kind"]
        assert len(seen) <= 366
        if year == 2026 and revision == 1:
            # Independent oracle transcribed from 国办发明电〔2025〕7号.
            rest = set()
            for start, end in [
                ("01-01", "01-03"), ("02-15", "02-23"), ("04-04", "04-06"),
                ("05-01", "05-05"), ("06-19", "06-21"), ("09-25", "09-27"), ("10-01", "10-07")
            ]:
                date, last = dt.date.fromisoformat(f"2026-{start}"), dt.date.fromisoformat(f"2026-{end}")
                while date <= last:
                    rest.add(date.isoformat())
                    date += dt.timedelta(days=1)
            work = {"2026-" + day for day in ["01-04", "02-14", "02-28", "05-09", "09-20", "10-10"]}
            assert {k for k, v in seen.items() if v == "rest"} == rest
            assert {k for k, v in seen.items() if v == "work"} == work
            assert len(rest) == 33 and len(work) == 6
        print(f"PASS: {year} revision {revision}, {len(seen)} explicit date overrides, digest verified")
    assert years, "Do not publish an empty calendar index"
    # Verify HTTP paths without publishing or contacting an external service.
    # This does not replace live GitHub Pages readback.
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(root)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for filename in ["index.json", *(item["file"] for item in index["years"])]:
            with urlopen(f"http://127.0.0.1:{server.server_port}/assets/holidays/cn/{filename}", timeout=5) as response:
                assert response.status == 200
                assert response.headers.get_content_type() == "application/json"
                assert response.read() == (folder / filename).read_bytes()
        print("PASS: local HTTP download paths and exact bytes")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, default=ROOT, help="Source or generated Jekyll root")
    args = parser.parse_args()
    validate(args.site)
