"""Dataset loading and writer-aware splitting.

The Mendeley *Potential Dysgraphia* dataset ships one scanned notebook page per
image and publishes no writer identifiers. ``load_dataset`` therefore derives a
subject key from the filename, which makes every image its own subject. The
split below honours real writer grouping when such a mapping is supplied and
reports loudly when it cannot, so that a reader is never left believing a
per-image split is a writer-independent one.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from .config import CONTROL_DIRNAME, DYSGRAPHIA_DIRNAME

try:  # only DysgraphiaDataset needs torch, and only once training starts
    from torch.utils.data import Dataset as _TorchDataset
except (ImportError, OSError):  # a missing or broken torch install must not block the split logic
    _TorchDataset = object

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
_SUBJECT_PATTERN = re.compile(r"\(([^)]*)\)")

Split = Tuple[np.ndarray, np.ndarray]


@dataclass
class SplitReport:
    """What the split actually did, recorded alongside the results."""

    n_images: int
    n_subjects: int
    subjects_unique_per_image: bool
    train_subjects: int
    val_subjects: int
    test_subjects: int
    train_samples: int
    val_samples: int
    test_samples: int
    train_labels: Dict[int, int]
    val_labels: Dict[int, int]
    test_labels: Dict[int, int]

    def as_text(self) -> str:
        lines = [
            f"images: {self.n_images}",
            f"subjects: {self.n_subjects}",
            f"subjects unique per image: {self.subjects_unique_per_image}",
            f"train: {self.train_samples} images / {self.train_subjects} subjects / labels {self.train_labels}",
            f"val:   {self.val_samples} images / {self.val_subjects} subjects / labels {self.val_labels}",
            f"test:  {self.test_samples} images / {self.test_subjects} subjects / labels {self.test_labels}",
        ]
        if self.subjects_unique_per_image:
            lines.append(
                "WARNING: every image resolved to a distinct subject, so this split is "
                "equivalent to a random per-image split. It is NOT writer-independent. "
                "Supply writer_map to load_dataset() to group pages by writer."
            )
        return "\n".join(lines)


class DysgraphiaDataset(_TorchDataset):
    """Yields ``(image, label, path)`` so downstream stages keep provenance."""

    def __init__(self, image_paths: Sequence[str], labels: Sequence[int], transform=None):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        import cv2

        path = self.image_paths[idx]
        image = cv2.imread(path)
        if image is None:
            raise IOError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx], path


def subject_key_from_filename(filename: str) -> str:
    """Extract the parenthesised index from names like ``LPD (12).jpg``."""
    match = _SUBJECT_PATTERN.search(filename)
    return match.group(1) if match else Path(filename).stem


def load_dataset(
    data_dir: Path | str,
    writer_map: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load both classes and return ``(image_paths, labels, subjects)``.

    Labels are ``0`` for low-potential dysgraphia (control) and ``1`` for
    potential dysgraphia. ``writer_map``, when supplied, maps an absolute image
    path to a real writer identifier; that is what makes the split genuinely
    writer-independent.
    """
    data_dir = Path(data_dir)
    image_paths: List[str] = []
    labels: List[int] = []
    subjects: List[str] = []

    for dirname, label, prefix in (
        (CONTROL_DIRNAME, 0, "LPD"),
        (DYSGRAPHIA_DIRNAME, 1, "PD"),
    ):
        class_dir = data_dir / dirname
        if not class_dir.is_dir():
            continue
        for filename in sorted(os.listdir(class_dir)):
            if not filename.lower().endswith(IMAGE_SUFFIXES):
                continue
            path = str(class_dir / filename)
            image_paths.append(path)
            labels.append(label)
            if writer_map and path in writer_map:
                subjects.append(writer_map[path])
            else:
                subjects.append(f"{prefix}_{subject_key_from_filename(filename)}")

    return np.array(image_paths), np.array(labels, dtype=int), np.array(subjects)


def writer_grouped_split(
    image_paths: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> Tuple[Split, Split, Split, SplitReport]:
    """Partition by subject so no writer appears in more than one split.

    Stratification uses each subject's majority label, the only label-level
    information a subject-grouped split can legitimately use.
    """
    unique_subjects = np.unique(subjects)
    subject_labels = np.array(
        [int(Counter(labels[subjects == s].tolist()).most_common(1)[0][0]) for s in unique_subjects]
    )

    unique_labels, label_counts = np.unique(subject_labels, return_counts=True)
    can_stratify = len(unique_labels) > 1 and label_counts.min() >= 2
    stratify_all = subject_labels if can_stratify else None

    train_val_subjects, test_subjects = train_test_split(
        unique_subjects,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_all,
    )

    if stratify_all is not None:
        keep = np.isin(unique_subjects, train_val_subjects)
        stratify_tv = subject_labels[keep]
        _, tv_counts = np.unique(stratify_tv, return_counts=True)
        if tv_counts.min() < 2:
            stratify_tv = None
    else:
        stratify_tv = None

    train_subjects, val_subjects = train_test_split(
        train_val_subjects,
        test_size=val_size / (1.0 - test_size),
        random_state=seed,
        stratify=stratify_tv,
    )

    def select(chosen: np.ndarray) -> Split:
        mask = np.isin(subjects, chosen)
        return image_paths[mask], labels[mask]

    train, val, test = select(train_subjects), select(val_subjects), select(test_subjects)

    report = SplitReport(
        n_images=int(len(image_paths)),
        n_subjects=int(len(unique_subjects)),
        subjects_unique_per_image=bool(len(unique_subjects) == len(image_paths)),
        train_subjects=int(len(train_subjects)),
        val_subjects=int(len(val_subjects)),
        test_subjects=int(len(test_subjects)),
        train_samples=int(len(train[0])),
        val_samples=int(len(val[0])),
        test_samples=int(len(test[0])),
        train_labels={int(k): int(v) for k, v in zip(*np.unique(train[1], return_counts=True))},
        val_labels={int(k): int(v) for k, v in zip(*np.unique(val[1], return_counts=True))},
        test_labels={int(k): int(v) for k, v in zip(*np.unique(test[1], return_counts=True))},
    )
    return train, val, test, report
