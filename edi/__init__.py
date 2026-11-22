"""Explainable Dysgraphia Index (EDI).

Reference implementation of the framework described in:

    V. Sreekumar, A. P. Anbumunee and S. Senthilkumar, "Image-Based Dysgraphia
    Detection with Explainability Using CAMs and EDI," 2026 International
    Conference on Computing, Communication, Control and Cyber-Physical Systems
    (I5CPS), 2026, pp. 1-6, doi: 10.1109/I5CPS67958.2026.11452617.

Typical use::

    from edi import PipelineConfig, run_pipeline

    outcome = run_pipeline(PipelineConfig(data_dir="data"), cam_method="scorecam")
    print(outcome["summary_table"])

Imports are lazy so that the evaluation and configuration layers, which need
only numpy, pandas and scikit-learn, stay usable in an environment without
PyTorch installed.
"""

from __future__ import annotations

from typing import Any

__version__ = "1.0.0"

_LAZY = {
    "PipelineConfig": ".config",
    "run_pipeline": ".pipeline",
    "DysgraphiaClassifier": ".model",
    "generate_scorecam": ".cam",
    "generate_gradcam": ".cam",
    "cam_stability_iou": ".cam",
    "load_dataset": ".data",
    "writer_grouped_split": ".data",
    "compute_handwriting_features": ".features",
    "compute_edi_features": ".features",
    "construct_edi": ".features",
    "evaluate_all": ".evaluate",
    "summarise": ".evaluate",
    "train_classifier": ".train",
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
