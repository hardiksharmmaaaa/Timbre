"""Global configuration loaded from environment / .env via pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """All runtime configuration. Overridable via env vars or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Reproducibility
    seed: int = 42
    log_level: str = "INFO"

    # Data locations (relative to repo root)
    data_root: Path = ROOT / "data"
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    splits_dir: Path = ROOT / "data" / "splits"
    model_dir: Path = ROOT / "models"
    experiment_dir: Path = ROOT / "experiments"

    # Feature extraction (frame semantics)
    frame_ms: int = 25
    stride_ms: int = 10
    n_mfcc: int = 20
    sample_rate: int = 22050

    # Reduction
    n_components: int = 8
    reduction_method: str = "pca"  # "pca" | "lle"

    # HMM
    hmm_n_states: int = 3  # latent acoustic regimes within each emotion-specific HMM
    hmm_n_states_emission_gmm: int = 1  # components per Gaussian mixture emission
    hmm_n_iter: int = 100

    # Bayesian layer
    bayesian_method: str = "vi"  # "vi" | "hybrid" | "mcmc"

    # Inference
    window_ms: int = Field(
        default=2000, validation_alias=AliasChoices("INFERENCE_WINDOW_MS", "WINDOW_MS")
    )
    latency_budget_ms: int = Field(
        default=300,
        validation_alias=AliasChoices("INFERENCE_LATENCY_BUDGET_MS", "LATENCY_BUDGET_MS"),
    )
    reset_on_silence: bool = True

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


settings = Settings()

# Ensure output dirs exist
for _d in (
    settings.processed_dir,
    settings.splits_dir,
    settings.model_dir,
    settings.experiment_dir,
):
    _d.mkdir(parents=True, exist_ok=True)
