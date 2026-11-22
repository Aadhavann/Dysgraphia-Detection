"""Tests for the data, metric and figure layers -- no PyTorch required.

These run against the real images under ``data/`` when they are present, and
against synthetic stand-ins otherwise, so the suite is useful in a bare CI
container as well as on a workstation that has the dataset.

    python tests/test_features.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from edi.config import RAW_FEATURE_NAMES  # noqa: E402
from edi.data import load_dataset, subject_key_from_filename, writer_grouped_split  # noqa: E402
from edi.features import compute_handwriting_features, construct_edi  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
HAS_IMAGES = DATA_DIR.is_dir() and any(DATA_DIR.rglob("*.jpg"))

EXPECTED_CONTROL = 135
EXPECTED_DYSGRAPHIA = 114


def _real_images(limit: int = 6) -> list[Path]:
    if not HAS_IMAGES:
        return []
    return sorted(DATA_DIR.rglob("*.jpg"))[:limit]


def _blank_image(size: int = 240) -> np.ndarray:
    """A page with no ink at all."""
    return np.full((size, size, 3), 255, dtype=np.uint8)


def _striped_image(size: int = 240) -> np.ndarray:
    """Synthetic handwriting-like marks on a white page."""
    rng = np.random.default_rng(0)
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    for row in range(40, size - 40, 45):
        for column in range(30, size - 30, 26):
            height = int(rng.integers(14, 30))
            width = int(rng.integers(6, 16))
            image[row : row + height, column : column + width] = rng.integers(0, 90)
    return image


def test_metrics_are_finite_and_non_negative() -> None:
    for image in (_striped_image(), _blank_image()):
        result = compute_handwriting_features(image)
        assert set(result.values) == set(RAW_FEATURE_NAMES)
        for name, value in result.values.items():
            assert np.isfinite(value), f"{name} is not finite"
            assert value >= 0.0, f"{name} is negative: {value}"


def test_blank_page_reports_nothing_measurable() -> None:
    """A blank page must be flagged, not silently reported as four zeros."""
    result = compute_handwriting_features(_blank_image())
    assert result.diagnostics["status"] == "insufficient_foreground"
    assert not any(result.diagnostics["measured"].values())


def test_marked_page_measures_every_metric() -> None:
    result = compute_handwriting_features(_striped_image())
    assert result.diagnostics["status"] == "ok"
    assert all(result.diagnostics["measured"].values()), result.diagnostics["measured"]
    assert result.values["spacing_irregularity"] > 0
    assert result.values["char_size_variance"] > 0
    assert result.values["stroke_width_std"] > 0


def test_cam_weighting_changes_the_metrics() -> None:
    """The CAM-weighted pass must differ from the whole-image pass."""
    image = _striped_image()
    cam = np.zeros(image.shape[:2], dtype=np.float32)
    cam[30:120, 30:150] = 1.0

    plain = compute_handwriting_features(image, cam_mask=None)
    weighted = compute_handwriting_features(image, cam_mask=cam)

    assert plain.diagnostics["cam_weighted"] is False
    assert weighted.diagnostics["cam_weighted"] is True
    assert plain.values != weighted.values, "CAM weighting had no effect at all"


def test_metrics_run_on_real_dataset_images() -> None:
    images = _real_images()
    if not images:
        print("    (skipped: dataset images not present)")
        return

    import cv2

    measured_counts = {name: 0 for name in RAW_FEATURE_NAMES}
    for path in images:
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        result = compute_handwriting_features(image)
        assert result.diagnostics["status"] == "ok", f"{path.name}: {result.diagnostics['status']}"
        for name, was_measured in result.diagnostics["measured"].items():
            measured_counts[name] += int(was_measured)
        for value in result.values.values():
            assert np.isfinite(value)

    assert len(images) >= 3
    for name, count in measured_counts.items():
        assert count > 0, f"{name} was never measurable across {len(images)} real images"


def test_subject_key_parsing() -> None:
    assert subject_key_from_filename("LPD (12).jpg") == "12"
    assert subject_key_from_filename("PD (7).jpeg") == "7"
    assert subject_key_from_filename("scan.png") == "scan"


def test_construct_edi_ignores_labels() -> None:
    """EDI standardisation must be unsupervised, or the index leaks the answer."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "label": rng.integers(0, 2, size=40),
            "cam_mean": rng.random(40),
        }
    )
    for name in RAW_FEATURE_NAMES:
        frame[f"{name}_cam"] = rng.normal(size=40)

    built, _ = construct_edi(frame)
    shuffled = frame.copy()
    shuffled["label"] = shuffled["label"].to_numpy()[::-1]
    rebuilt, _ = construct_edi(shuffled)

    assert "EDI_equal" in built and "EDI_weighted" in built
    np.testing.assert_allclose(built["EDI_equal"], rebuilt["EDI_equal"])
    np.testing.assert_allclose(built["EDI_weighted"], rebuilt["EDI_weighted"])


def test_construct_edi_handles_nan() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"label": rng.integers(0, 2, size=20), "cam_mean": rng.random(20)})
    for name in RAW_FEATURE_NAMES:
        frame[f"{name}_cam"] = rng.normal(size=20)
    frame.loc[0, "stroke_width_std_cam"] = np.nan

    built, _ = construct_edi(frame)
    assert built["EDI_equal"].notna().all()
    assert built["EDI_weighted"].notna().all()


def test_load_dataset_counts_match_the_published_dataset() -> None:
    if not HAS_IMAGES:
        print("    (skipped: dataset images not present)")
        return
    paths, labels, subjects = load_dataset(DATA_DIR)
    assert len(paths) == EXPECTED_CONTROL + EXPECTED_DYSGRAPHIA, f"found {len(paths)} images"
    assert int((labels == 0).sum()) == EXPECTED_CONTROL
    assert int((labels == 1).sum()) == EXPECTED_DYSGRAPHIA
    assert len(np.unique(subjects)) == len(paths)


def test_split_is_disjoint_complete_and_honest_about_subjects() -> None:
    if not HAS_IMAGES:
        print("    (skipped: dataset images not present)")
        return
    paths, labels, subjects = load_dataset(DATA_DIR)
    train, val, test, report = writer_grouped_split(paths, labels, subjects, seed=42)

    train_set, val_set, test_set = set(train[0]), set(val[0]), set(test[0])
    assert not (train_set & val_set) and not (train_set & test_set) and not (val_set & test_set)
    assert len(train_set) + len(val_set) + len(test_set) == len(paths)

    # The dataset ships no writer identifiers, so the split must say so.
    assert report.subjects_unique_per_image is True
    assert "NOT writer-independent" in report.as_text()

    for partition in (train, val, test):
        assert set(np.unique(partition[1])) <= {0, 1}
        assert len(np.unique(partition[1])) == 2, "a partition lost one of the two classes"


def test_split_is_reproducible_for_a_fixed_seed() -> None:
    if not HAS_IMAGES:
        print("    (skipped: dataset images not present)")
        return
    paths, labels, subjects = load_dataset(DATA_DIR)
    first = writer_grouped_split(paths, labels, subjects, seed=42)
    second = writer_grouped_split(paths, labels, subjects, seed=42)
    third = writer_grouped_split(paths, labels, subjects, seed=7)

    assert set(first[0][0]) == set(second[0][0])
    assert set(first[0][0]) != set(third[0][0]), "changing the seed did not change the split"


def test_figures_render_from_a_feature_table() -> None:
    """Catches layout and column errors in every non-CAM figure."""
    import tempfile

    from edi.visualize import (
        plot_correlation_matrix,
        plot_edi_distributions,
        plot_feature_distributions,
        plot_roc_curves,
        plot_training_curves,
    )
    from edi.evaluate import evaluate_all

    rng = np.random.default_rng(2)
    n = 50
    labels = np.array([0] * 25 + [1] * 25)
    rng.shuffle(labels)
    frame = pd.DataFrame(
        {
            "label": labels,
            "cnn_prob": np.clip(labels * 0.55 + 0.25 + rng.normal(scale=0.28, size=n), 0.001, 0.999),
            "cam_mean": rng.random(n),
        }
    )
    for name in RAW_FEATURE_NAMES:
        frame[f"{name}_raw"] = rng.normal(size=n)
        frame[f"{name}_cam"] = labels * 0.9 + rng.normal(size=n)
    frame, _ = construct_edi(frame)

    history = types.SimpleNamespace(
        train_loss=list(np.linspace(0.7, 0.05, 12)),
        val_loss=list(np.linspace(0.7, 0.25, 12)),
        train_acc=list(np.linspace(0.6, 1.0, 12)),
        val_acc=list(np.linspace(0.6, 0.92, 12)),
        val_auc=list(np.linspace(0.65, 0.98, 12)),
        learning_rate=[1e-3] * 12,
        best_val_auc=0.98,
        best_epoch=9,
        stopped_epoch=-1,
        epochs_run=12,
    )

    results = evaluate_all(frame, cv_folds=5, seed=42)

    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary)
        for call in (
            lambda d: plot_training_curves(history, d),
            lambda d: plot_feature_distributions(frame, d),
            lambda d: plot_edi_distributions(frame, d),
            lambda d: plot_correlation_matrix(frame, d),
            lambda d: plot_roc_curves(frame, results, d),
        ):
            path = call(target)
            assert path.exists() and path.stat().st_size > 1000, f"{path.name} was not written"


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as error:
            failures += 1
            print(f"FAIL  {test.__name__}: {error}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
