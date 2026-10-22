"""Canonical six-class dataset manifest and speaker-disjoint split preparation.

The manifest is the contract shared by feature extraction, training, calibration, and
evaluation. RAVDESS and CREMA-D are development data. TESS is always external-test data.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import soundfile as sf

from ml.config import EmotionState, get_logger, parse_ravdess_fields

log = get_logger("manifest")

SIX_CLASS_STATES = (
    EmotionState.NEUTRAL,
    EmotionState.HAPPY,
    EmotionState.SAD,
    EmotionState.ANGRY,
    EmotionState.FEARFUL,
    EmotionState.DISGUST,
)

CREMA_LABELS = {
    "NEU": EmotionState.NEUTRAL,
    "HAP": EmotionState.HAPPY,
    "SAD": EmotionState.SAD,
    "ANG": EmotionState.ANGRY,
    "FEA": EmotionState.FEARFUL,
    "DIS": EmotionState.DISGUST,
}
TESS_LABELS = {
    "neutral": EmotionState.NEUTRAL,
    "happy": EmotionState.HAPPY,
    "sad": EmotionState.SAD,
    "angry": EmotionState.ANGRY,
    "fear": EmotionState.FEARFUL,
    "disgust": EmotionState.DISGUST,
}
DATASET_METADATA = {
    "RAVDESS": {
        "role": "development",
        "license": "CC BY-NC-SA 4.0",
        "url": "https://zenodo.org/records/1188976",
    },
    "CREMA-D": {
        "role": "development",
        "license": "ODbL 1.0 / DbCL 1.0",
        "url": "https://github.com/CheyneyComputerScience/CREMA-D",
    },
    "TESS": {
        "role": "external_test",
        "license": "TESS dataset terms",
        "url": "https://doi.org/10.5683/SP2/E8H2MF",
    },
}

MANIFEST_FIELDS = (
    "clip_id",
    "dataset",
    "relative_audio_path",
    "speaker_id",
    "canonical_label",
    "label_id",
    "original_label",
    "sample_rate",
    "duration_seconds",
    "split",
    "role",
    "intensity",
    "annotation_agreement",
    "source_license",
    "source_url",
    "audio_sha256",
    "channels",
    "rms",
    "clipping_rate",
    "silence_fraction",
)

EXCLUSION_FIELDS = (
    "dataset",
    "relative_audio_path",
    "clip_id",
    "reason",
    "audio_sha256",
)


@dataclass(frozen=True)
class ParsedClip:
    path: Path
    dataset: str
    speaker_id: str
    label: EmotionState
    original_label: str
    intensity: str = ""
    annotation_agreement: float | None = None

    @property
    def role(self) -> str:
        return DATASET_METADATA[self.dataset]["role"]

    @property
    def clip_id(self) -> str:
        prefix = self.dataset.lower().replace("-", "_")
        return f"{prefix}__{self.path.stem}"


@dataclass(frozen=True)
class ManifestRecord:
    clip_id: str
    dataset: str
    relative_audio_path: str
    speaker_id: str
    canonical_label: str
    label_id: int
    original_label: str
    sample_rate: int
    duration_seconds: float
    split: str
    role: str
    intensity: str
    annotation_agreement: float | None
    source_license: str
    source_url: str
    audio_sha256: str
    channels: int
    rms: float
    clipping_rate: float
    silence_fraction: float
    exclusion_reason: str = ""

    @property
    def accepted(self) -> bool:
        return not self.exclusion_reason

    def manifest_row(self) -> dict:
        row = asdict(self)
        row.pop("exclusion_reason")
        return row

    def exclusion_row(self) -> dict:
        return {
            "dataset": self.dataset,
            "relative_audio_path": self.relative_audio_path,
            "clip_id": self.clip_id,
            "reason": self.exclusion_reason,
            "audio_sha256": self.audio_sha256,
        }


def parse_ravdess_clip(path: Path) -> ParsedClip | None:
    """Parse one RAVDESS speech filename, excluding non-shared labels."""
    fields = parse_ravdess_fields(path.name)
    modality, vocal_channel, emotion, intensity, _, _, actor = fields
    if (modality, vocal_channel) != (3, 1):
        return None
    state = EmotionState(emotion)
    if state not in SIX_CLASS_STATES:
        return None
    return ParsedClip(
        path=path,
        dataset="RAVDESS",
        speaker_id=f"RAVDESS:{actor:02d}",
        label=state,
        original_label=state.name_lower,
        intensity={1: "normal", 2: "strong"}[intensity],
    )


def parse_crema_clip(path: Path, agreement: float | None = None) -> ParsedClip:
    """Parse one CREMA-D filename into the canonical six-class contract."""
    fields = path.stem.split("_")
    if len(fields) != 4:
        raise ValueError(f"invalid CREMA-D filename: {path.name!r}")
    actor, sentence, label_code, intensity = fields
    if not actor.isdigit() or len(actor) != 4 or len(sentence) != 3:
        raise ValueError(f"invalid CREMA-D filename: {path.name!r}")
    try:
        state = CREMA_LABELS[label_code]
    except KeyError as exc:
        raise ValueError(f"invalid CREMA-D label: {path.name!r}") from exc
    if intensity not in {"LO", "MD", "HI", "XX"}:
        raise ValueError(f"invalid CREMA-D intensity: {path.name!r}")
    return ParsedClip(
        path=path,
        dataset="CREMA-D",
        speaker_id=f"CREMA-D:{actor}",
        label=state,
        original_label=label_code,
        intensity=intensity,
        annotation_agreement=agreement,
    )


def parse_tess_clip(path: Path) -> ParsedClip | None:
    """Parse one TESS filename; pleasant-surprise clips are out of six-class scope."""
    fields = path.stem.split("_")
    if len(fields) < 3:
        raise ValueError(f"invalid TESS filename: {path.name!r}")
    speaker, label_code = fields[0], fields[-1].lower()
    if speaker not in {"OAF", "YAF"}:
        raise ValueError(f"invalid TESS speaker: {path.name!r}")
    if label_code == "ps":
        return None
    try:
        state = TESS_LABELS[label_code]
    except KeyError as exc:
        raise ValueError(f"invalid TESS label: {path.name!r}") from exc
    return ParsedClip(
        path=path,
        dataset="TESS",
        speaker_id=f"TESS:{speaker}",
        label=state,
        original_label=label_code,
    )


def load_crema_agreement(path: Path) -> dict[str, float]:
    """Load normalized voice-vote agreement from CREMA-D's summary table."""
    if not path.exists():
        return {}
    agreement: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            stem = (row.get("FileName") or "").strip()
            raw_level = (row.get("VoiceLevel") or "").strip()
            if not stem or not raw_level:
                continue
            try:
                agreement[stem] = float(raw_level) / 100.0
            except ValueError:
                continue
    return agreement


def discover_clips(raw_dir: Path) -> tuple[list[ParsedClip], dict]:
    """Discover every six-class candidate and report excluded-label/parser counts."""
    candidates: list[ParsedClip] = []
    summary: dict[str, dict[str, int]] = {}
    agreement = load_crema_agreement(
        raw_dir / "CREMA-D" / "processedResults" / "summaryTable.csv"
    )
    specs = (
        ("RAVDESS", raw_dir / "RAVDESS", parse_ravdess_clip),
        (
            "CREMA-D",
            raw_dir / "CREMA-D" / "AudioWAV",
            lambda path: parse_crema_clip(path, agreement.get(path.stem)),
        ),
        ("TESS", raw_dir / "TESS", parse_tess_clip),
    )
    for dataset, root, parser in specs:
        counts = Counter()
        for path in sorted(root.rglob("*.wav")):
            counts["discovered"] += 1
            try:
                clip = parser(path)
            except ValueError:
                counts["invalid_filename"] += 1
                continue
            if clip is None:
                counts["outside_six_class_scope"] += 1
                continue
            candidates.append(clip)
            counts["six_class_candidates"] += 1
        summary[dataset] = dict(counts)
    return candidates, summary


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_silence_fraction(y: np.ndarray, sample_rate: int) -> float:
    frame_length = max(1, round(sample_rate * 0.025))
    padding = (-y.size) % frame_length
    if padding:
        y = np.pad(y, (0, padding))
    frames = y.reshape(-1, frame_length)
    frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    return float(np.mean(frame_rms < 1e-4))


def screen_clip(parsed: ParsedClip, raw_dir: Path) -> ManifestRecord:
    """Decode, hash, and quality-screen one candidate without modifying source audio."""
    metadata = DATASET_METADATA[parsed.dataset]
    relative = parsed.path.relative_to(raw_dir).as_posix()
    digest = ""
    reason = ""
    sample_rate = 0
    channels = 0
    duration = 0.0
    rms = 0.0
    clipping_rate = 0.0
    silence_fraction = 1.0
    try:
        digest = file_sha256(parsed.path)
        audio, sample_rate = sf.read(parsed.path, dtype="float32", always_2d=True)
        channels = int(audio.shape[1]) if audio.ndim == 2 else 1
        if audio.size == 0 or sample_rate <= 0:
            reason = "empty_audio"
        elif not np.isfinite(audio).all():
            reason = "non_finite_audio"
        else:
            mono = np.mean(audio, axis=1, dtype=np.float64)
            duration = float(mono.size / sample_rate)
            rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
            clipping_rate = float(np.mean(np.abs(mono) >= 0.999))
            silence_fraction = _frame_silence_fraction(mono, sample_rate)
            if duration < 0.2:
                reason = "too_short"
            elif rms < 1e-4 or silence_fraction >= 0.98:
                reason = "nearly_silent"
            elif clipping_rate > 0.05:
                reason = "severely_clipped"
    except Exception as exc:  # noqa: BLE001 - exclusion reason belongs in the manifest
        reason = f"unreadable_audio:{type(exc).__name__}"

    return ManifestRecord(
        clip_id=parsed.clip_id,
        dataset=parsed.dataset,
        relative_audio_path=relative,
        speaker_id=parsed.speaker_id,
        canonical_label=parsed.label.name_lower,
        label_id=int(parsed.label),
        original_label=parsed.original_label,
        sample_rate=int(sample_rate),
        duration_seconds=round(duration, 6),
        split="",
        role=parsed.role,
        intensity=parsed.intensity,
        annotation_agreement=parsed.annotation_agreement,
        source_license=metadata["license"],
        source_url=metadata["url"],
        audio_sha256=digest,
        channels=channels,
        rms=round(rms, 9),
        clipping_rate=round(clipping_rate, 9),
        silence_fraction=round(silence_fraction, 9),
        exclusion_reason=reason,
    )


def reject_exact_duplicates(records: list[ManifestRecord]) -> list[ManifestRecord]:
    """Reject conflicting-label copies; retain one deterministic same-label copy."""
    output = list(records)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.accepted and record.audio_sha256:
            groups[record.audio_sha256].append(index)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        labels = {records[index].label_id for index in indices}
        if len(labels) > 1:
            label_names = ",".join(sorted({records[index].canonical_label for index in indices}))
            for index in indices:
                output[index] = replace(
                    records[index],
                    exclusion_reason=f"conflicting_labels_for_identical_audio:{label_names}",
                )
            continue
        first = records[indices[0]].clip_id
        for index in indices[1:]:
            output[index] = replace(
                records[index], exclusion_reason=f"exact_duplicate_of:{first}"
            )
    return output


def _partition_speakers(
    speakers: list[str], seed: int, val_fraction: float, test_fraction: float
) -> dict[str, set[str]]:
    if len(speakers) < 3:
        raise ValueError("development datasets need at least three speakers")
    values = np.array(sorted(speakers), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    n_test = min(max(1, round(values.size * test_fraction)), values.size - 2)
    n_val = min(max(1, round(values.size * val_fraction)), values.size - n_test - 1)
    return {
        "test": set(values[:n_test].tolist()),
        "val": set(values[n_test:n_test + n_val].tolist()),
        "train": set(values[n_test + n_val:].tolist()),
    }


def assign_splits(
    records: list[ManifestRecord],
    seed: int,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[list[ManifestRecord], dict]:
    """Assign independent per-dataset speaker splits; lock TESS to external test."""
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("split fractions must be positive and sum to less than one")
    by_dataset: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        if record.accepted:
            by_dataset[record.dataset].append(record)

    assignments: dict[str, str] = {}
    split_speakers: dict[str, dict[str, list[str]]] = {}
    for dataset in ("RAVDESS", "CREMA-D"):
        dataset_records = by_dataset.get(dataset, [])
        speakers = sorted({record.speaker_id for record in dataset_records})
        dataset_seed = seed + sum(dataset.encode("utf-8"))
        partitions = _partition_speakers(speakers, dataset_seed, val_fraction, test_fraction)
        split_speakers[dataset] = {
            split: sorted(values) for split, values in partitions.items()
        }
        for record in dataset_records:
            matches = [split for split, group in partitions.items() if record.speaker_id in group]
            if len(matches) != 1:
                raise RuntimeError(f"speaker split assignment failed for {record.clip_id}")
            assignments[record.clip_id] = matches[0]

    tess_speakers = sorted({record.speaker_id for record in by_dataset.get("TESS", [])})
    split_speakers["TESS"] = {"external_test": tess_speakers}
    for record in by_dataset.get("TESS", []):
        assignments[record.clip_id] = "external_test"

    output = [
        replace(record, split=assignments.get(record.clip_id, "excluded"))
        for record in records
    ]
    return output, split_speakers


def _csv_bytes(records: list[dict], fields: tuple[str, ...]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def prepare_manifest(
    raw_dir: Path,
    manifest_dir: Path,
    splits_dir: Path,
    seed: int = 42,
    workers: int = 4,
) -> dict:
    """Build the complete manifest, exclusions, frozen splits, hashes, and summary."""
    candidates, discovery = discover_clips(raw_dir)
    log.info("quality-screening %d six-class candidates", len(candidates))
    if workers <= 1:
        records = [screen_clip(clip, raw_dir) for clip in candidates]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(lambda clip: screen_clip(clip, raw_dir), candidates))
    records = reject_exact_duplicates(records)
    records, split_speakers = assign_splits(records, seed)

    accepted = sorted((record for record in records if record.accepted), key=lambda r: r.clip_id)
    excluded = sorted(
        (record for record in records if not record.accepted), key=lambda r: r.clip_id
    )
    manifest_payload = _csv_bytes([record.manifest_row() for record in accepted], MANIFEST_FIELDS)
    manifest_hash = hashlib.sha256(manifest_payload).hexdigest()
    manifest_path = manifest_dir / "all_clips.csv"
    exclusion_path = manifest_dir / "exclusions.csv"
    _atomic_write(manifest_path, manifest_payload)
    _atomic_write(
        exclusion_path,
        _csv_bytes([record.exclusion_row() for record in excluded], EXCLUSION_FIELDS),
    )
    _atomic_write(manifest_dir / "all_clips.sha256", f"{manifest_hash}\n".encode())

    split_clips: dict[str, list[str]] = defaultdict(list)
    for record in accepted:
        split_clips[record.split].append(record.clip_id)
    split_document = {
        "version": "six_class_v1",
        "seed": seed,
        "strategy": "speaker-disjoint-within-development-dataset",
        "manifest_sha256": manifest_hash,
        "speakers": split_speakers,
        "clips": {name: sorted(ids) for name, ids in sorted(split_clips.items())},
    }
    _atomic_write(
        splits_dir / "six_class_v1.json",
        (json.dumps(split_document, indent=2, sort_keys=True) + "\n").encode(),
    )

    by_dataset = Counter(record.dataset for record in accepted)
    by_split = Counter(record.split for record in accepted)
    by_label = Counter(record.canonical_label for record in accepted)
    exclusion_reasons = Counter(record.exclusion_reason for record in excluded)
    summary = {
        "version": "six_class_v1",
        "seed": seed,
        "manifest_sha256": manifest_hash,
        "discovery": discovery,
        "accepted_clips": len(accepted),
        "excluded_clips": len(excluded),
        "accepted_by_dataset": dict(sorted(by_dataset.items())),
        "accepted_by_split": dict(sorted(by_split.items())),
        "accepted_by_label": dict(sorted(by_label.items())),
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "quality_thresholds": {
            "minimum_duration_seconds": 0.2,
            "minimum_rms": 1e-4,
            "maximum_silence_fraction": 0.98,
            "maximum_clipping_rate": 0.05,
        },
    }
    _atomic_write(
        manifest_dir / "summary.json",
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
    )
    return summary


def read_manifest(path: Path) -> list[dict]:
    """Read the canonical CSV with stable numeric types for downstream stages."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label_id"] = int(row["label_id"])
        row["sample_rate"] = int(row["sample_rate"])
        row["duration_seconds"] = float(row["duration_seconds"])
        row["annotation_agreement"] = (
            float(row["annotation_agreement"]) if row["annotation_agreement"] else None
        )
    return rows
