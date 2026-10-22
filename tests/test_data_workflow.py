"""Regression tests for dataset preparation using isolated local fixtures."""

import io
import zipfile

import pytest

from ml.config import EmotionState
from ml.data import download as download_data
from ml.training import extract_features


def test_discovery_accepts_actor_directories_and_skips_unrelated_files(tmp_path):
    actor_dir = tmp_path / "Actor_01"
    actor_dir.mkdir()
    speech = actor_dir / "03-01-04-01-01-01-01.wav"
    speech.touch()
    (actor_dir / "03-02-04-01-01-01-01.wav").touch()
    (actor_dir / "notes.wav").touch()
    (actor_dir / "03-01-99-01-01-01-01.wav").touch()

    assert extract_features.find_ravdess_clips(tmp_path) == [(speech, EmotionState.SAD)]


def test_discovery_skips_malformed_names_in_legacy_layout(tmp_path):
    legacy_dir = tmp_path / "Audio_Speech_Actors_01-24"
    legacy_dir.mkdir()
    (legacy_dir / "notes.wav").touch()

    assert extract_features.find_ravdess_clips(tmp_path) == []


def test_download_extracts_an_existing_archive(tmp_path, monkeypatch):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("Actor_01/clip.wav", b"fixture")

    def unexpected_download(*args, **kwargs):
        pytest.fail("an existing valid archive must not be downloaded again")

    monkeypatch.setattr(download_data.requests, "get", unexpected_download)
    download_data.download_and_unzip("https://example.test/dataset.zip", tmp_path)

    assert (tmp_path / "Actor_01" / "clip.wav").read_bytes() == b"fixture"


def test_interrupted_download_does_not_publish_an_archive(tmp_path, monkeypatch):
    class InterruptedResponse:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"partial archive"
            raise ConnectionError("download interrupted")

        def close(self):
            pass

    monkeypatch.setattr(download_data.requests, "get", lambda *a, **k: InterruptedResponse())

    with pytest.raises(ConnectionError, match="interrupted"):
        download_data.download_and_unzip("https://example.test/dataset.zip", tmp_path)

    assert not (tmp_path / "dataset.zip").exists()


@pytest.mark.parametrize("cached_contents", [None, b"interrupted old archive"])
def test_download_publishes_and_extracts_complete_archive(tmp_path, monkeypatch, cached_contents):
    archive = tmp_path / "dataset.zip"
    if cached_contents is not None:
        archive.write_bytes(cached_contents)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as output:
        output.writestr("Actor_01/clip.wav", b"fixture")

    class CompleteResponse:
        closed = False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield buffer.getvalue()

        def close(self):
            self.closed = True

    response = CompleteResponse()
    monkeypatch.setattr(download_data.requests, "get", lambda *a, **k: response)

    assert download_data.download_and_unzip("https://example.test/dataset.zip", tmp_path) == archive
    assert zipfile.is_zipfile(archive)
    assert (tmp_path / "Actor_01" / "clip.wav").read_bytes() == b"fixture"
    assert response.closed
    assert sorted(path.name for path in tmp_path.iterdir()) == ["Actor_01", "dataset.zip"]
