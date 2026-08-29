"""Download RAVDESS (and optionally TESS/CREMA-D) into data/raw.

Command: python -m scripts.download_data
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests

from timbre.utils.common import get_logger
from timbre.utils.config import settings

log = get_logger("download")


def download_and_unzip(url: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1].split("?")[0]
    local = dest / name
    if local.exists() and local.stat().st_size > 0:
        log.info("already present: %s", local)
        return local
    log.info("downloading %s", url)
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(local, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
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
