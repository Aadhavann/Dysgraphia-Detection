"""Held-out evaluation of every head in the comparison.

The single most important property here is that **no head is ever scored on a
label it was fitted to.** The earlier revision of this pipeline called
``fit(X, y_true)`` and then ``score(y_true, predict(X))`` on the same rows, so
every reported figure was an in-sample fit; with two features and fifty rows
that reaches a perfect 1.000 AUC regardless of whether the features carry
signal. Everything below is evaluated out-of-fold instead, and the sample size
is small enough that a confidence interval is reported alongside each point
estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RAW_FEATURE_NAMES

CNN_PROBABILITY_COLUMN = "cnn_prob"
POSITIVE_THRESHOLD = 0.5


@dataclass
class HeadResult:
    """Metrics for one head, plus how those metrics were produced."""

    name: str
    protocol: str
    accuracy: float
    auc: float
    auc_ci_low: Optional[float]
    auc_ci_high: Optional[float]
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: List[List[int]] = field(default_factory=list)
    n_samples: int = 0
    n_positive: int = 0
    cv_folds: Optional[int] = None
    note: str = ""

    def as_row(self) -> Dict[str, object]:
        return {
            "head": self.name,
            "protocol": self.protocol,
            "accuracy": round(self.accuracy, 4),
            "auc": round(self.auc, 4),
            "auc_ci": (
                f"[{self.auc_ci_low:.3f}, {self.auc_ci_high:.3f}]"
                if self.auc_ci_low is not None and self.auc_ci_high is not None
                else "n/a"
            ),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "n": self.n_samples,
        }


def _head_feature_columns(name: str) -> List[str]:
    if name == "features_only":
        return [f"{metric}_raw" for metric in RAW_FEATURE_NAMES]
    if name == "cam_features_only":
        return [f"{metric}_cam" for metric in RAW_FEATURE_NAMES]
    if name == "edi_equal":
        return ["EDI_equal"]
    if name == "edi_weighted":
        return ["EDI_weighted"]
    if name == "cnn_edi_fusion":
        return [CNN_PROBABILITY_COLUMN, "EDI_weighted"]
    raise ValueError(f"Unknown head: {name!r}")


HEAD_NAMES = (
    "cnn_only",
    "features_only",
    "cam_features_only",
    "edi_equal",
    "edi_weighted",
    "cnn_edi_fusion",
)


def _feasible_folds(y: np.ndarray, requested: int) -> int:
    """Largest stratified k such that every class has at least k members."""
    _, counts = np.unique(y, return_counts=True)
    return max(2, min(requested, int(counts.min())))


def mann_whitney_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """AUC via the Mann-Whitney U statistic, using average ranks for ties.

    Equivalent to ``roc_auc_score`` but expressed so that many bootstrap
    replicates can be ranked in one vectorised call.
    """
    positive = y_true == 1
    n_pos = int(positive.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(y_score)
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstraps: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI for the AUC.

    With a test partition of roughly fifty images the AUC is not stable to two
    decimal places, and quoting one without an interval overstates precision.
    Resampling is drawn as one index matrix so the whole bootstrap is a handful
    of array operations rather than thousands of metric calls.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n = y_true.size
    if n == 0 or np.unique(y_true).size < 2:
        return None, None

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstraps, n))
    resampled_true = y_true[indices] == 1
    resampled_score = y_score[indices]

    n_pos = resampled_true.sum(axis=1)
    n_neg = n - n_pos
    usable = (n_pos > 0) & (n_neg > 0)
    if not usable.any():
        return None, None

    ranks = rankdata(resampled_score[usable], axis=1)
    positive_rank_sums = np.where(resampled_true[usable], ranks, 0.0).sum(axis=1)
    aucs = (positive_rank_sums - n_pos[usable] * (n_pos[usable] + 1) / 2.0) / (
        n_pos[usable] * n_neg[usable]
    )

    return float(np.percentile(aucs, 100 * alpha / 2)), float(np.percentile(aucs, 100 * (1 - alpha / 2)))


def _metrics_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    y_pred = (y_score >= POSITIVE_THRESHOLD).astype(int)
    zero_division = 0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=zero_division)),
        "recall": float(recall_score(y_true, y_pred, zero_division=zero_division)),
        "f1": float(f1_score(y_true, y_pred, zero_division=zero_division)),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _make_head() -> Pipeline:
    """A small regularised linear head, scaled inside the CV pipeline.

    Scaling lives inside the pipeline so each fold's scaler is fitted on that
    fold's training rows only.
    """
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
        ]
    )


def evaluate_cnn_only(features_df: pd.DataFrame) -> HeadResult:
    """Score the CNN directly from its stored probabilities. Nothing is fitted."""
    y_true = features_df["label"].to_numpy(dtype=int)
    y_score = features_df[CNN_PROBABILITY_COLUMN].to_numpy(dtype=float)

    metrics = _metrics_from_scores(y_true, y_score)
    low, high = bootstrap_auc_ci(y_true, y_score)

    return HeadResult(
        name="cnn_only",
        protocol="direct: stored test-set softmax probabilities, no fitting",
        auc=float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        auc_ci_low=low,
        auc_ci_high=high,
        n_samples=int(len(y_true)),
        n_positive=int(y_true.sum()),
        **metrics,
    )


def evaluate_fitted_head(features_df: pd.DataFrame, name: str, cv_folds: int = 5, seed: int = 42) -> HeadResult:
    """Evaluate a linear head with out-of-fold predictions.

    Every probability used for scoring comes from a fold in which that row was
    held out, so the reported AUC is an honest generalisation estimate for the
    head rather than a measure of how well it memorised fifty rows.
    """
    columns = _head_feature_columns(name)
    missing = [column for column in columns if column not in features_df.columns]
    if missing:
        raise KeyError(f"head {name!r} needs column(s) {missing} which are not in the feature table")

    X = features_df[columns].to_numpy(dtype=float)
    y_true = features_df["label"].to_numpy(dtype=int)

    folds = _feasible_folds(y_true, cv_folds)
    note = "" if folds == cv_folds else f"cv_folds reduced from {cv_folds} to {folds}: minority class too small"

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    y_score = cross_val_predict(_make_head(), X, y_true, cv=splitter, method="predict_proba")[:, 1]

    metrics = _metrics_from_scores(y_true, y_score)
    low, high = bootstrap_auc_ci(y_true, y_score)

    return HeadResult(
        name=name,
        protocol=f"out-of-fold: StratifiedKFold({folds}) cross_val_predict",
        auc=float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        auc_ci_low=low,
        auc_ci_high=high,
        n_samples=int(len(y_true)),
        n_positive=int(y_true.sum()),
        cv_folds=folds,
        note=note,
        **metrics,
    )


def evaluate_all(features_df: pd.DataFrame, cv_folds: int = 5, seed: int = 42) -> Dict[str, HeadResult]:
    """Run every head on the same held-out partition and return the results."""
    if CNN_PROBABILITY_COLUMN not in features_df.columns:
        raise KeyError(
            f"feature table has no {CNN_PROBABILITY_COLUMN!r} column; recompute features with the current pipeline"
        )

    results: Dict[str, HeadResult] = {"cnn_only": evaluate_cnn_only(features_df)}
    for name in HEAD_NAMES:
        if name == "cnn_only":
            continue
        results[name] = evaluate_fitted_head(features_df, name, cv_folds=cv_folds, seed=seed)
    return results


def summarise(results: Dict[str, HeadResult]) -> pd.DataFrame:
    """A tidy table of every head, suitable for the README and for CSV export."""
    return pd.DataFrame([result.as_row() for result in results.values()])


def fusion_improves_on_cnn(results: Dict[str, HeadResult]) -> Optional[float]:
    """AUC gained by adding the EDI to the CNN, or None if either is missing."""
    cnn, fusion = results.get("cnn_only"), results.get("cnn_edi_fusion")
    if cnn is None or fusion is None:
        return None
    if np.isnan(cnn.auc) or np.isnan(fusion.auc):
        return None
    return float(fusion.auc - cnn.auc)


def format_report(results: Dict[str, HeadResult], split_text: str = "", stability: Optional[Dict[str, object]] = None) -> str:
    """Human-readable summary written to results/results_summary.txt."""
    lines = ["Explainable Dysgraphia Index - evaluation report", "=" * 64, ""]

    if split_text:
        lines += ["Data partition", "-" * 64, split_text, ""]

    lines += [
        "Evaluation protocol",
        "-" * 64,
        "All figures are computed on the held-out test partition.",
        "cnn_only uses the CNN's stored softmax probabilities and fits nothing.",
        "Every other head is fitted with StratifiedKFold cross_val_predict, so no",
        "head is scored against a label it was fitted to.",
        "AUC confidence intervals are 95% percentile bootstrap over 2000 resamples.",
        "",
        "Results",
        "-" * 64,
    ]

    table = summarise(results)
    lines.append(table.to_string(index=False))
    lines.append("")

    delta = fusion_improves_on_cnn(results)
    if delta is None:
        lines.append("Fusion vs CNN: not computable")
    else:
        verdict = "improves" if delta > 0 else "does not improve"
        lines.append(f"Fusion vs CNN-only AUC: {delta:+.4f} ({verdict})")
    lines.append("")

    if stability is not None:
        lines += ["CAM stability", "-" * 64]
        mean_iou = stability.get("mean_iou")
        if mean_iou is None:
            lines.append(f"not measured ({stability.get('note', 'insufficient models')})")
        else:
            lines.append(
                f"mean IoU across {stability.get('n_pairs')} model pair(s) on "
                f"{stability.get('n_images')} image(s): {mean_iou:.3f} "
                f"(sd {stability.get('std_iou'):.3f}, method {stability.get('method')}, "
                f"threshold {stability.get('threshold')})"
            )
        lines.append("")

    return "\n".join(lines)
