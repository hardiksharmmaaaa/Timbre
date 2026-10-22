"""Download RAVDESS (and optionally TESS/CREMA-D) into data/raw.

Command: python -m ml.data.download
"""

from __future__ import annotations

import zipfile
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import requests

from ml.config import get_logger, settings

log = get_logger("download")


def download_and_unzip(url: str, dest: Path) -> Path:
    """Reuse complete ZIP archives and publish new downloads only after validation."""
    dest.mkdir(parents=True, exist_ok=True)
    name = Path(urlsplit(url).path).name
    if not name:
        raise ValueError("dataset URL must include an archive filename")
    local = dest / name
    if local.is_file() and zipfile.is_zipfile(local):
        log.info("already present: %s", local)
    else:
        log.info("downloading archive %s", name)
        with TemporaryDirectory(dir=dest, prefix="download-") as staging_dir:
            pending = Path(staging_dir) / name
            with closing(requests.get(url, stream=True, timeout=120)) as response:
                response.raise_for_status()
                with pending.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        output.write(chunk)
            with zipfile.ZipFile(pending) as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise zipfile.BadZipFile(f"archive checksum failed for {bad_member}")
            pending.replace(local)
    with zipfile.ZipFile(local) as z:
        z.extractall(dest)
    log.info("extracted to %s", dest)
    return local


def main() -> None:
    if settings.ravdess_url:
        download_and_unzip(settings.ravdess_url, settings.raw_dir / "RAVDESS")
    if settings.crema_url:
        download_and_unzip(settings.crema_url, settings.raw_dir / "CREMA-D")
    if settings.tess_url:
        download_and_unzip(settings.tess_url, settings.raw_dir / "TESS")


if __name__ == "__main__":
    main()
