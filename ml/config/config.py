"""Global configuration loaded from environment / .env via pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# This module lives at <repo>/ml/config/config.py; keep all runtime artifacts
# rooted in the repository after the ML package move.
ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime configuration. Overridable via env vars or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Reproducibility
    seed: int = 42
    log_level: str = "INFO"

    # Data locations (relative to repo root)
    data_root: Path = ROOT / "data"
    raw_dir: Path = ROOT / "data" / "raw"
    manifest_dir: Path = ROOT / "data" / "manifests"
    processed_dir: Path = ROOT / "data" / "processed"
    augmentation_dir: Path = ROOT / "data" / "processed_augmented"
    splits_dir: Path = ROOT / "data" / "splits"
    model_dir: Path = ROOT / "models"
    experiment_dir: Path = ROOT / "experiments"

    # Feature extraction (frame semantics)
    frame_ms: int = Field(default=25, gt=0)
    stride_ms: int = Field(default=10, gt=0)
    n_mfcc: int = Field(default=20, gt=0)
    sample_rate: int = Field(default=16000, gt=0)
    f0_frame_ms: int = Field(default=50, gt=0)
    f0_min_hz: float = Field(default=50.0, gt=0)
    f0_max_hz: float = Field(default=500.0, gt=0)

    # Reduction
    n_components: int = Field(default=8, gt=0)
    reduction_method: str = "pca"  # "pca" | "lle"

    # HMM
    hmm_n_states: int = Field(default=3, gt=0)  # acoustic regimes per emotion-specific HMM
    hmm_n_states_emission_gmm: int = Field(default=1, gt=0)  # Gaussian mixture components
    hmm_n_iter: int = Field(default=100, gt=0)

    # Bayesian layer
    bayesian_method: str = "vi"  # "vi" | "hybrid" | "mcmc"

    # Inference
    window_ms: int = Field(
        default=2000, gt=0, validation_alias=AliasChoices("INFERENCE_WINDOW_MS", "WINDOW_MS")
    )
    latency_budget_ms: int = Field(
        default=300,
        gt=0,
        validation_alias=AliasChoices("INFERENCE_LATENCY_BUDGET_MS", "LATENCY_BUDGET_MS"),
    )
    reset_on_silence: bool = True
    # Audio below this RMS level is treated as silence and abstained from.
    # Mirrors the frontend's VAD threshold (VAD_THRESHOLD) for the same 0-1 scale.
    silence_rms_threshold: float = Field(
        default=0.005,
        ge=0.0,
        validation_alias=AliasChoices("SILENCE_RMS_THRESHOLD", "SILENCE_THRESHOLD"),
    )
    # Which trained model serves live inference: the per-emotion HMM ("hmm") or the
    # clip-statistic logistic baseline ("clip"). The baseline scores better on the
    # held-out in-domain set (49.9% accuracy vs 39.0%) and is the recommended default.
    served_model: str = Field(
        default="clip", validation_alias=AliasChoices("SERVED_MODEL", "SERVED_MODEL_NAME")
    )
    # Clip-classifier path abstains when the top probability falls below this value.
    # Applies only when served_model == "clip" (the HMM uses calibration.json instead).
    baseline_abstain_confidence: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("BASELINE_ABSTAIN_CONFIDENCE", "ABSTAIN_CONFIDENCE"),
    )

    # External services
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    tts_provider: str = ""
    tts_api_key: str = ""
    tts_voice_id: str = ""

    # Datasets
    ravdess_url: str = ""
    crema_url: str = ""
    tess_url: str = ""

    @model_validator(mode="after")
    def validate_feature_bounds(self) -> Settings:
        if self.f0_min_hz >= self.f0_max_hz:
            raise ValueError("F0_MIN_HZ must be less than F0_MAX_HZ")
        if self.f0_max_hz >= self.sample_rate / 2:
            raise ValueError("F0_MAX_HZ must be below the Nyquist frequency")
        return self


settings = Settings()

# Ensure output dirs exist
for _d in (
    settings.manifest_dir,
    settings.processed_dir,
    settings.augmentation_dir,
    settings.splits_dir,
    settings.model_dir,
    settings.experiment_dir,
):
    _d.mkdir(parents=True, exist_ok=True)
