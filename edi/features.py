"""Handwriting metrics, CAM-weighted extraction, and EDI construction.

The four metrics follow the occupational-therapy criteria the paper adopts:
spacing irregularity, baseline wobble, character size variance and stroke width
variation. Each is computed twice, once on the whole image and once on the
CAM-weighted image, and the pair is what makes the index interpretable: the
difference between the two says whether the model's attention landed on the
regions that actually carry the signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

from .config import RAW_FEATURE_NAMES

if TYPE_CHECKING:  # torch is only needed for the CAM pass, not for the metrics
    import torch

    from .model import DysgraphiaClassifier

MIN_FOREGROUND_PIXELS = 100
MIN_CONTOUR_AREA_SPACING = 10
MIN_CONTOUR_AREA_SIZE = 20
MIN_CENTROIDS = 3
MIN_BASELINE_PEAKS = 3


@dataclass
class FeatureResult:
    values: Dict[str, float]
    diagnostics: Dict[str, object] = field(default_factory=dict)


def _empty_features() -> Dict[str, float]:
    return {name: 0.0 for name in RAW_FEATURE_NAMES}


def compute_handwriting_features(
    image: np.ndarray,
    cam_mask: Optional[np.ndarray] = None,
) -> FeatureResult:
    """Compute the four metrics, optionally on a CAM-weighted image.

    A metric stays at ``0.0`` when the image does not support measuring it (too
    few foreground pixels, fewer than three connected components, and so on).
    The returned diagnostics record which metrics were actually measurable, so
    a zero can be told apart from a genuine measurement of zero.
    """
    import cv2

    features = _empty_features()
    measured = {name: False for name in RAW_FEATURE_NAMES}

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()

    if cam_mask is not None:
        cam_resized = cv2.resize(cam_mask.astype(np.float32), (gray.shape[1], gray.shape[0]))
        gray = (gray.astype(np.float32) * cam_resized).astype(np.uint8)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    foreground = int((binary > 0).sum())

    diagnostics: Dict[str, object] = {
        "cam_weighted": cam_mask is not None,
        "foreground_pixels": foreground,
        "measured": measured,
    }

    if foreground < MIN_FOREGROUND_PIXELS:
        diagnostics["status"] = "insufficient_foreground"
        return FeatureResult(features, diagnostics)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Spacing irregularity: spread of horizontal gaps between component centroids.
    centroids = []
    for contour in contours:
        if cv2.contourArea(contour) <= MIN_CONTOUR_AREA_SPACING:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            centroids.append((int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])))

    if len(centroids) >= MIN_CENTROIDS:
        centroids.sort(key=lambda point: point[0])
        distances = [centroids[i + 1][0] - centroids[i][0] for i in range(len(centroids) - 1)]
        if len(distances) > 1:
            features["spacing_irregularity"] = float(np.std(distances))
            measured["spacing_irregularity"] = True

    # Baseline wobble: residual variance of text-line peak positions about a
    # straight-line fit. Matches the estimator used in the published work.
    projection = np.sum(binary, axis=1)
    if projection.size and projection.max() > 0:
        peaks, _ = find_peaks(projection, height=projection.max() * 0.1)
        if len(peaks) > MIN_BASELINE_PEAKS:
            positions = np.arange(len(peaks), dtype=float)
            coefficients = np.polyfit(positions, peaks.astype(float), 1)
            fitted = np.polyval(coefficients, positions)
            features["baseline_wobble"] = float(np.var(peaks.astype(float) - fitted))
            measured["baseline_wobble"] = True

    # Character size variance: mean of bounding-box height and width variance.
    heights, widths = [], []
    for contour in contours:
        if cv2.contourArea(contour) <= MIN_CONTOUR_AREA_SIZE:
            continue
        _, _, width, height = cv2.boundingRect(contour)
        heights.append(height)
        widths.append(width)

    if len(heights) > 1:
        features["char_size_variance"] = float((np.var(heights) + np.var(widths)) / 2.0)
        measured["char_size_variance"] = True

    # Stroke width variation: spread of the distance transform over ink pixels.
    distance_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    stroke_widths = distance_transform[binary > 0] * 2.0
    if stroke_widths.size > 1:
        features["stroke_width_std"] = float(np.std(stroke_widths))
        measured["stroke_width_std"] = True

    diagnostics["status"] = "ok"
    return FeatureResult(features, diagnostics)


def compute_edi_features(
    model: "DysgraphiaClassifier",
    dataloader,
    device: "torch.device",
    cam_method: str = "scorecam",
    num_channels: int = 32,
    progress_every: int = 10,
) -> pd.DataFrame:
    """Build the per-image feature table for one partition.

    Each row carries the CNN's predicted class *and* its softmax probability for
    the positive class, which is what makes a real ROC-AUC and a real late
    fusion possible. Earlier revisions stored only the hard label and then
    treated it as a probability, which makes the CNN's AUC a restatement of its
    balanced accuracy rather than a ranking metric.
    """
    import torch

    from .cam import generate_cam

    model.eval()
    rows: List[Dict[str, object]] = []
    failures: List[str] = []

    total_batches = len(dataloader)
    with torch.no_grad():
        for batch_index, (images, labels, paths) in enumerate(dataloader):
            if progress_every and (batch_index + 1) % progress_every == 0:
                print(f"  features: batch {batch_index + 1}/{total_batches}")

            for i in range(images.shape[0]):
                path = paths[i]
                try:
                    image = images[i : i + 1].to(device)
                    logits = model(image)
                    probability = torch.softmax(logits, dim=1)
                    predicted = int(probability.argmax(dim=1).item())

                    cam = generate_cam(
                        model, image, predicted, device, method=cam_method, num_channels=num_channels
                    )
                    if cam is None:
                        failures.append(f"{path}: CAM unavailable")
                        continue
                    cam_np = cam.detach().cpu().numpy().squeeze()

                    # Undo the ImageNet normalisation to recover display-range
                    # pixels; the metrics are calibrated on 0-255 grey levels.
                    raw = images[i].permute(1, 2, 0).numpy()
                    raw_uint8 = (np.clip(raw, 0.0, 1.0) * 255).astype(np.uint8)

                    plain = compute_handwriting_features(raw_uint8, cam_mask=None)
                    weighted = compute_handwriting_features(raw_uint8, cam_mask=cam_np)

                    row: Dict[str, object] = {
                        "path": path,
                        "label": int(labels[i].item()),
                        "prediction": predicted,
                        "cnn_prob": float(probability[0, 1].item()),
                        "cam_mean": float(np.mean(cam_np)),
                        "cam_std": float(np.std(cam_np)),
                        "cam_max": float(np.max(cam_np)),
                        "cam_method": cam_method,
                    }
                    for name in RAW_FEATURE_NAMES:
                        row[f"{name}_raw"] = plain.values[name]
                        row[f"{name}_cam"] = weighted.values[name]
                        row[f"{name}_raw_measured"] = plain.diagnostics["measured"][name]
                        row[f"{name}_cam_measured"] = weighted.diagnostics["measured"][name]

                    rows.append(row)
                except Exception as error:  # one unreadable image must not end the run
                    failures.append(f"{path}: {error}")
                    continue

    frame = pd.DataFrame(rows)
    if failures:
        print(f"  features: {len(failures)} image(s) skipped")
        for entry in failures[:5]:
            print(f"    - {entry}")
    return frame


def construct_edi(features_df: pd.DataFrame, low_variance_eps: float = 1e-6) -> Tuple[pd.DataFrame, StandardScaler]:
    """Add ``EDI_equal`` and ``EDI_weighted`` columns.

    Standardisation is unsupervised: the scaler sees feature values only, never
    labels, so fitting it on the partition being evaluated leaks no label
    information. Both index variants are reported because they answer different
    questions -- the equal-weight index is the interpretable clinical score,
    while the CAM-weighted index is the one the fusion model consumes.
    """
    cam_columns = [f"{name}_cam" for name in RAW_FEATURE_NAMES]
    frame = features_df.copy()

    for column in cam_columns:
        missing = int(frame[column].isna().sum())
        if missing:
            print(f"  EDI: {column} had {missing} NaN value(s); filled with 0")
            frame[column] = frame[column].fillna(0.0)

    scaler = StandardScaler()
    normalised = scaler.fit_transform(frame[cam_columns].values)

    for index, column in enumerate(cam_columns):
        if float(np.std(frame[column].values)) < low_variance_eps:
            print(f"  EDI WARNING: {column} has near-zero variance and carries no signal")

    frame["EDI_equal"] = normalised.mean(axis=1)

    cam_weights = frame["cam_mean"].values.reshape(-1, 1)
    weighted_sum = (normalised * cam_weights).sum(axis=1)
    weight_total = cam_weights.sum(axis=1) + 1e-8
    frame["EDI_weighted"] = weighted_sum / weight_total

    return frame, scaler
