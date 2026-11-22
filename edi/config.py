"""Central configuration for the Explainable Dysgraphia Index pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224

CONTROL_DIRNAME = "Low Potential Dysgraphia"
DYSGRAPHIA_DIRNAME = "Potential Dysgraphia"

RAW_FEATURE_NAMES = (
    "spacing_irregularity",
    "baseline_wobble",
    "char_size_variance",
    "stroke_width_std",
)

FEATURE_DIMS = {"resnet18": 512, "efficientnet_b0": 1280}


@dataclass(frozen=True)
class PipelineConfig:
    """Every tunable in one place so a run is fully described by its config."""

    data_dir: Path = REPO_ROOT / "data"
    results_dir: Path = REPO_ROOT / "results"

    model_type: str = "resnet18"
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 10

    test_size: float = 0.2
    val_size: float = 0.2
    seed: int = 42

    scorecam_channels: int = 32
    cam_threshold: float = 0.5
    stability_runs: int = 2

    cv_folds: int = 5
    low_variance_eps: float = 1e-6
