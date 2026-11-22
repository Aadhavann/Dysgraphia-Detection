"""Figures for the paper and the repository.

Output filenames match the ones already committed under ``results/`` so the
site and the README keep working: ``training_curves.png``,
``feature_distributions.png``, ``edi_distributions.png``,
``correlation_matrix.png`` and ``case_study_*.png``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from .config import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, RAW_FEATURE_NAMES
from .evaluate import HeadResult

if TYPE_CHECKING:  # torch-dependent modules are imported only where needed
    import torch

    from .model import DysgraphiaClassifier
    from .train import TrainingHistory

PRETTY_NAMES = {
    "spacing_irregularity": "Spacing irregularity",
    "baseline_wobble": "Baseline wobble",
    "char_size_variance": "Character size variance",
    "stroke_width_std": "Stroke width variation",
}

CLASS_NAMES = {0: "Low potential", 1: "Potential dysgraphia"}

FIGURE_NAMES = (
    "training_curves.png",
    "feature_distributions.png",
    "edi_distributions.png",
    "correlation_matrix.png",
    "roc_curves.png",
)


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Composite a normalised heatmap onto an image for qualitative inspection."""
    import cv2

    resized = cv2.resize(cam.astype(np.float32), (image_rgb.shape[1], image_rgb.shape[0]))
    heat = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (image_rgb.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8)


def _evaluation_transform():
    import torchvision.transforms as transforms

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )


def plot_training_curves(history: TrainingHistory, save_dir: Path | str) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = np.arange(1, history.epochs_run + 1)

    axes[0].plot(epochs, history.train_loss, label="train")
    axes[0].plot(epochs, history.val_loss, label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history.train_acc, label="train")
    axes[1].plot(epochs, history.val_acc, label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    axes[2].plot(epochs, history.val_auc, color="tab:green")
    if history.best_epoch > 0:
        axes[2].axvline(history.best_epoch, linestyle="--", linewidth=1, color="gray",
                        label=f"best (epoch {history.best_epoch})")
        axes[2].legend()
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Validation AUC")
    axes[2].set_title("Model selection metric")

    fig.tight_layout()
    path = save_dir / "training_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_distributions(features_df: pd.DataFrame, save_dir: Path | str) -> Path:
    """Raw metrics on the top row, CAM-weighted on the bottom."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, len(RAW_FEATURE_NAMES), figsize=(4 * len(RAW_FEATURE_NAMES), 7))
    for row, suffix in enumerate(("raw", "cam")):
        for column_index, metric in enumerate(RAW_FEATURE_NAMES):
            axis = axes[row, column_index]
            for label in (0, 1):
                values = features_df.loc[features_df["label"] == label, f"{metric}_{suffix}"]
                axis.hist(values, bins=20, alpha=0.6, label=CLASS_NAMES[label])
            title = PRETTY_NAMES[metric] + (" (CAM-weighted)" if suffix == "cam" else "")
            axis.set_title(title, fontsize=10)
            axis.set_xlabel("Value", fontsize=8)
            if column_index == 0:
                axis.set_ylabel("Count", fontsize=8)
            if row == 0 and column_index == 0:
                axis.legend(fontsize=7)

    fig.tight_layout()
    path = save_dir / "feature_distributions.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_edi_distributions(features_df: pd.DataFrame, save_dir: Path | str) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, column in zip(axes, ("EDI_equal", "EDI_weighted")):
        for label in (0, 1):
            values = features_df.loc[features_df["label"] == label, column]
            axis.hist(values, bins=20, alpha=0.6, label=CLASS_NAMES[label])
        axis.set_title("EDI, equal weights" if column == "EDI_equal" else "EDI, CAM-weighted")
        axis.set_xlabel("Index score")
        axis.set_ylabel("Count")
        axis.legend(fontsize=8)

    fig.tight_layout()
    path = save_dir / "edi_distributions.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_correlation_matrix(features_df: pd.DataFrame, save_dir: Path | str) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    columns = [f"{metric}_{suffix}" for suffix in ("raw", "cam") for metric in RAW_FEATURE_NAMES]
    columns += ["EDI_equal", "EDI_weighted", "cam_mean", "cnn_prob"]
    columns = [column for column in columns if column in features_df.columns]

    labels = [PRETTY_NAMES.get(column.rsplit("_", 1)[0], column) + (
        " (CAM)" if column.endswith("_cam") else " (raw)" if column.endswith("_raw") else ""
    ) for column in columns]

    fig, axis = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        features_df[columns].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 7},
        ax=axis,
    )
    axis.set_title("Feature correlation")
    axis.tick_params(axis="x", rotation=45, labelsize=8)
    axis.tick_params(axis="y", rotation=0, labelsize=8)

    fig.tight_layout()
    path = save_dir / "correlation_matrix.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curves(features_df: pd.DataFrame, results: Dict[str, HeadResult], save_dir: Path | str) -> Path:
    """ROC curves for the heads that produce a continuous score."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    y_true = features_df["label"].to_numpy(dtype=int)
    _, counts = np.unique(y_true, return_counts=True)
    folds = max(2, min(5, int(counts.min())))

    def head(columns: Sequence[str]):
        X = features_df[list(columns)].to_numpy(dtype=float)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=42))])
        return cross_val_predict(model, X, y_true, cv=splitter, method="predict_proba")[:, 1]

    curves = {
        "CNN only": features_df["cnn_prob"].to_numpy(dtype=float),
        "EDI, equal weights": head(["EDI_equal"]),
        "EDI, CAM-weighted": head(["EDI_weighted"]),
        "CNN + EDI fusion": head(["cnn_prob", "EDI_weighted"]),
    }

    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    for label, scores in curves.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        axis.plot(fpr, tpr, linewidth=1.6, label=f"{label} (AUC {roc_auc_score(y_true, scores):.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(f"Test partition, n = {len(y_true)}")
    axis.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    path = save_dir / "roc_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_case_studies(
    model: "DysgraphiaClassifier",
    features_df: pd.DataFrame,
    device: "torch.device",
    save_dir: Path | str,
    cam_method: str = "scorecam",
    num_channels: int = 32,
) -> List[Path]:
    """One figure per outcome class, showing attention next to the metrics.

    The false-positive and false-negative panels matter as much as the correct
    ones: they are where a clinician can see the explanation disagree with the
    prediction, which is the whole point of publishing an interpretable index.
    """
    import torch  # noqa: F401  (kept explicit: tensors are handled below)

    from .cam import generate_cam

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    transform = _evaluation_transform()

    buckets = {
        "true_positive": (features_df["label"] == 1) & (features_df["prediction"] == 1),
        "true_negative": (features_df["label"] == 0) & (features_df["prediction"] == 0),
        "false_positive": (features_df["label"] == 0) & (features_df["prediction"] == 1),
        "false_negative": (features_df["label"] == 1) & (features_df["prediction"] == 0),
    }

    written: List[Path] = []
    for case_type, mask in buckets.items():
        if not mask.any():
            print(f"  case study skipped: no {case_type.replace('_', ' ')} in the test partition")
            continue
        row = features_df.loc[mask].iloc[0]

        import cv2

        image_bgr = cv2.imread(str(row["path"]))
        if image_bgr is None:
            print(f"  case study skipped: unreadable image for {case_type}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        tensor = transform(image_rgb).unsqueeze(0).to(device)
        cam = generate_cam(model, tensor, int(row["prediction"]), device, method=cam_method, num_channels=num_channels)
        if cam is None:
            print(f"  case study skipped: CAM unavailable for {case_type}")
            continue
        cam_np = cam.detach().cpu().numpy().squeeze()

        fig = plt.figure(figsize=(15, 9))
        grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.28, wspace=0.22)

        axis_original = fig.add_subplot(grid[0, 0])
        axis_original.imshow(image_rgb)
        axis_original.set_title("Handwriting sample")
        axis_original.axis("off")

        axis_heat = fig.add_subplot(grid[0, 1])
        heat = axis_heat.imshow(cam_np, cmap="jet")
        axis_heat.set_title(f"{cam_method.upper()} activation")
        axis_heat.axis("off")
        fig.colorbar(heat, ax=axis_heat, fraction=0.046, pad=0.04)

        axis_overlay = fig.add_subplot(grid[0, 2])
        axis_overlay.imshow(overlay_heatmap(image_rgb, cam_np))
        axis_overlay.set_title("Overlay")
        axis_overlay.axis("off")

        axis_features = fig.add_subplot(grid[1, 0])
        positions = np.arange(len(RAW_FEATURE_NAMES))
        width = 0.36
        raw_values = [float(row[f"{metric}_raw"]) for metric in RAW_FEATURE_NAMES]
        cam_values = [float(row[f"{metric}_cam"]) for metric in RAW_FEATURE_NAMES]
        axis_features.bar(positions - width / 2, raw_values, width, label="whole image", alpha=0.8)
        axis_features.bar(positions + width / 2, cam_values, width, label="CAM-weighted", alpha=0.8)
        axis_features.set_xticks(positions)
        axis_features.set_xticklabels([PRETTY_NAMES[m].replace(" ", "\n") for m in RAW_FEATURE_NAMES], fontsize=7)
        axis_features.set_title("Metric values")
        axis_features.legend(fontsize=7)

        axis_edi = fig.add_subplot(grid[1, 1])
        edi_values = [float(row["EDI_equal"]), float(row["EDI_weighted"])]
        axis_edi.bar(["equal", "CAM-weighted"], edi_values, alpha=0.8,
                     color=["tab:green" if v <= 0 else "tab:red" for v in edi_values])
        axis_edi.axhline(0, color="gray", linewidth=0.8)
        axis_edi.set_title("EDI score")
        axis_edi.set_ylabel("Index value")

        axis_info = fig.add_subplot(grid[1, 2])
        axis_info.axis("off")
        info = (
            f"Outcome:        {case_type.replace('_', ' ')}\n"
            f"True label:     {CLASS_NAMES[int(row['label'])]}\n"
            f"Prediction:     {CLASS_NAMES[int(row['prediction'])]}\n"
            f"CNN p(dysg.):   {float(row['cnn_prob']):.3f}\n\n"
            f"CAM mean:       {float(row['cam_mean']):.3f}\n"
            f"CAM std:        {float(row['cam_std']):.3f}\n"
            f"CAM max:        {float(row['cam_max']):.3f}\n\n"
            f"EDI equal:      {float(row['EDI_equal']):+.3f}\n"
            f"EDI weighted:   {float(row['EDI_weighted']):+.3f}"
        )
        axis_info.text(0.0, 1.0, info, transform=axis_info.transAxes, va="top", ha="left",
                       fontsize=9, family="monospace")
        axis_info.set_title("Case detail")

        fig.suptitle(f"Case study: {case_type.replace('_', ' ')}", fontsize=13)
        path = save_dir / f"case_study_{case_type}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    return written


def write_all_figures(
    features_df: pd.DataFrame,
    history: TrainingHistory,
    results: Dict[str, HeadResult],
    save_dir: Path | str,
) -> Dict[str, Path]:
    """Every non-case-study figure in one call."""
    save_dir = Path(save_dir)
    return {
        "training_curves": plot_training_curves(history, save_dir),
        "feature_distributions": plot_feature_distributions(features_df, save_dir),
        "edi_distributions": plot_edi_distributions(features_df, save_dir),
        "correlation_matrix": plot_correlation_matrix(features_df, save_dir),
        "roc_curves": plot_roc_curves(features_df, results, save_dir),
    }
