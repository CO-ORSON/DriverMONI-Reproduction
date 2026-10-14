"""Streaming, checksummed acquisition for large public input snapshots."""

from contextlib import contextmanager
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zipfile import ZipFile

from .common import sha256_file, utc_now, write_json


class FileCache:
    """Cache remote bytes or explicit local inputs, with optional source locks."""

    def __init__(self, root, *, offline=False, refresh=False, lock=None):
        if offline and refresh:
            raise ValueError("offline and refresh are mutually exclusive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.offline, self.refresh = offline, refresh
        self.lock = None if lock is None else {r["url"]: r["sha256"] for r in lock["sources"]}
        self.sources = {}
        self.last_kegg_request = 0.0

    def get_text(self, url):
        path = self.get(url, {"url": url, "format": "text"})
        return path.read_text(encoding="utf-8-sig")

    def get(self, role, spec):
        url = Path(spec["local_path"]).expanduser().resolve().as_uri() if "local_path" in spec else spec["url"]
        if self.lock is not None and url not in self.lock:
            raise ValueError(f"Source is absent from lock: {url}")
        if "local_path" in spec:
            path = Path(spec["local_path"]).expanduser().resolve()
            digest = sha256_file(path)
            metadata = {"url": url, "sha256": digest, "bytes": path.stat().st_size,
                        "acquisition": "explicit_local_input"}
        else:
            key = hashlib.sha256(url.encode()).hexdigest()
            path, meta = self.root / (key + ".bin"), self.root / (key + ".json")
            if path.exists() and meta.exists() and not self.refresh:
                metadata = json.loads(meta.read_text())
                if metadata.get("url") != url or sha256_file(path) != metadata.get("sha256"):
                    raise ValueError(f"Cached source failed checksum verification: {url}")
            else:
                if self.offline:
                    raise FileNotFoundError(f"No verified cached source for {url}")
                metadata = self._download(url, path, expected_sha=None if self.lock is None else self.lock[url])
                write_json(meta, metadata)
        if self.lock is not None and metadata["sha256"] != self.lock[url]:
            raise ValueError(f"Source differs from lock: {url}")
        self.sources[role] = dict(metadata, role=role, specification=spec)
        return path

    def _download(self, url, destination, *, expected_sha=None):
        temporary = destination.with_suffix(".download")
        for attempt in range(3):
            if urlparse(url).hostname == "rest.kegg.jp":
                time.sleep(max(0, 0.4 - (time.monotonic() - self.last_kegg_request)))
                self.last_kegg_request = time.monotonic()
            digest, size = hashlib.sha256(), 0
            request = Request(url, headers={"User-Agent": "DriverMONI-source-preparation/1.0"})
            try:
                with urlopen(request, timeout=45) as response, temporary.open("wb") as handle:
                    if "html" in response.headers.get("Content-Type", "").lower():
                        raise ValueError(f"Received HTML instead of raw data: {url}")
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        size += len(chunk)
                        if size > 2 * 1024**3:
                            raise ValueError("Source exceeds 2 GiB acquisition limit")
                        digest.update(chunk)
                        handle.write(chunk)
                    if size == 0:
                        raise ValueError(f"Empty source: {url}")
                    metadata = {"url": url, "resolved_url": response.geturl(), "sha256": digest.hexdigest(),
                                "bytes": size, "retrieved_at": utc_now(), "acquisition": "download",
                                "last_modified": response.headers.get("Last-Modified"),
                                "etag": response.headers.get("ETag")}
                if expected_sha is not None and metadata["sha256"] != expected_sha:
                    raise ValueError(f"Downloaded source differs from lock: {url}")
                temporary.replace(destination)
                return metadata
            except HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise
            except (URLError, TimeoutError, ConnectionError):
                if attempt == 2:
                    raise
            time.sleep(2**attempt)
        raise RuntimeError("Source acquisition did not complete")


@contextmanager
def open_text_source(path, spec):
    """Stream text, gzip, or an explicitly selected ZIP member without extraction."""
    kind = spec["format"]
    if kind == "zip":
        with ZipFile(path) as archive, archive.open(spec["member"]) as binary:
            with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as handle:
                yield handle
    elif kind == "gzip":
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            yield handle
    elif kind == "text":
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            yield handle
    else:
        raise ValueError(f"Unsupported text-source format: {kind}")
